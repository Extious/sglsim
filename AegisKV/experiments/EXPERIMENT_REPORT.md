# AegisKV 全局虚拟时钟论文实验设计与历史聚合结果

更新时间：2026-08-29

项目：`/home/comp/zhanzhao/sglsim`，分支 `aegiskv`

历史 aggregate lineage：[`20260829T064719Z_paper_complete_v2`](results/20260829T064719Z_paper_complete_v2/)

当前权威 GlobalDES P0 合同 lineage：
[`20260829T112907Z_global_des_p0_contract_v4/P0`](results/20260829T112907Z_global_des_p0_contract_v4/P0/)

设计语义：`source_owned_independent_hbm_dram_v2`

本报告只讨论 **SGLSim 模拟实验**，不设计、不汇总、也不要求任何 Mooncake/真实 GPU 实验。
目标执行引擎必须使用单一全局事件序，执行 SGLang-aligned continuous batching，并输出每个 DP rank、
每张物理 GPU HBM、每节点 DRAM 和每个 KV transfer 的完整时间线。完整实现合同见
[`GLOBAL_VIRTUAL_CLOCK_DESIGN.md`](GLOBAL_VIRTUAL_CLOCK_DESIGN.md)。

> **证据状态纠正：** 历史 613-cell lineage 使用 `AegisCompleteSimulator` 的 aggregate metadata-DES，
> 不是上述 GPU/resource-level 模拟器。因此其 10 张图全部降级为历史探索性结果，不能作为目标论文
> 实验的当前结果。当前 P0 v4 **contract gate 已执行 12/12 cells 并全部 PASS**，且
> `target_first_and_node_failure` 已调用 current controller + bridge rotation 闭环；但权威 summary 仍明确记录
> `contract_execution_only=true`、`full_scheduler_integration=false`、
> `p1_p12_execution_ready=false` 和 `performance_result=false`。真实 Offline Scheduler 的 P1/P4/P11/P12
> 只有小型 smoke；其中 P4 artifact 是修复 request-completion/eviction extent lifetime 释放链之前生成，
> 不能升级为正确性 gate 或论文数据。该 release chain 现已在代码中修复并通过 154-test 联合回归；
> 第一次 post-fix P4 尝试 `104732` 又暴露“7 completed + 1 scheduled 却被 suite 标记 complete”的请求终态
> 聚合 bug，因此仍没有有效的 post-fix Scheduler-integration lineage。P11 `102845` 只执行一个完整
> 57-request small workflow 的 recompute smoke。三层 JSON 可机械展开为 `12 + 2,134 = 2,146` cells，带显式
> frozen modeled bindings 时 deployment/failure runtime compiler 覆盖 P1--P12 的 2,134/2,134 cells；
> 这只证明配置可编译到运行时。完整 suite 未运行，正式论文性能结果仍为 **0 cells**，
> `paper_ready=false`。

## 0. 当前目标与完成状态

| 能力 | 当前状态 | 论文门槛 |
| --- | --- | --- |
| 单 cell 全局事件堆 | GlobalDES P0 v4：**12/12 PASS** | 继续保持稳定 priority/sequence 和确定性 event hash |
| SGLang continuous batching | `OfflineSchedulerCluster` 已接入真实 SGLSim Scheduler worker 和单一 GlobalDES；extent release 代码修复后 154-test 回归通过 | 修复 request terminal aggregation，再生成新的 immutable Scheduler-integration gate lineage；104732 无效 |
| 每 DP rank 状态 | prepare/complete、arrival、queue/batch 与 resource snapshot 已进入全局事件序 | 正式 lineage 必须落盘完整 `rank_steps` 并通过 bounded-online accuracy audit |
| 每物理 GPU HBM | Scheduler snapshot 经 topology bridge 展开到 `(node,gpu,dp,pp,tp)`，weights/runtime/primary/prefix/Aegis pending/committed 分账已接通 | 元数据账本不是实际 HBM 分配；正式 cells 仍需容量与 conservation audit |
| 每节点 DRAM | node-local HiCache/Aegis pending/committed pool、node fail/empty repair 已接通 | 不是实际 DRAM 占用或带宽测量 |
| KV 迁移资源竞争 | protection 和 recovery jobs 使用 copy engine/PCIe/NVLink/NIC/fabric/DRAM lane 日历 | 带宽/setup 仍为 modeled input，尚无论文 lineage |
| 独立 tier commit | HBM/DRAM 各自只在全部 TP shard transfer-complete 后 notify/commit；一方 abort 不终止另一方 | 必须在新 gate lineage 中保留 group/ACK 证据 |
| hard fail-stop/recovery | 支持 compute cancel、worker terminate、volatile loss、node-local DRAM loss、empty restart、Peer-HBM/DRAM recovery job 和 replay | 仍缺单物理 GPU/process 的 TP-shard 级 fail-stop、full-Scheduler target-before-source route rotation、committed output position 的 silent replay 和 recovery `TransferRecord` |
| Request 配置 | TraceLab/P6 causal workload 已编译；P11 observed Weka artifact 已生成 30 workflows/13,918 rows | P11 simultaneous coverage 未达 0.95；readiness/输入产物不等于执行 |
| Causal Slack 输入 | TraceLab 顺序背景显式 zero-ticket；P4 controlled fork/join 有 admission-visible modeled ticket；P11 使用 request-return-only cross-fit ticket；P12 只允许 analytical ticket | P11 safe Slack LCB 非零数为 0；不得用 post-hoc completion 重算 Slack |
| Runtime compiler | 带 frozen model/modeled-timing/simulator/failure bindings 时 P1--P12 **2,134/2,134 可编译** | 只验证 deployment/failure/runtime projection，不验证 workload 执行、生命周期正确性或性能 |
| P1--P12 runner | real-Scheduler runner/check/resume 与 fail-close 框架已有单测；已有 P1/P12 各1-request、P4 8-request、P11 完整 57-request workflow smoke | smoke 均为 modeled/preliminary/non-paper-ready；P4 101529 是 pre-fix artifact，104732 有终态聚合 bug；完整正式结果仍 0 cells |

### 0.1 证据台账：什么已通过，什么没有跑

| 证据 | 当前结果 | 允许的 claim | 禁止的 claim |
| --- | --- | --- | --- |
| P0 v4 immutable lineage | 12/12 contract cells PASS；5 组 v4 诊断图；target-first cell 的 4/4 assertions 与 10/10 invariants PASS | GlobalDES 合同与 current controller + bridge target-before-source rotation 在冻结 deterministic scenario 中通过 | 不得声称 full Scheduler integration、P1--P12 ready 或 performance result |
| GlobalDES/Scheduler/resource/failure 代码回归 | targeted tests | 当前代码级接入有回归覆盖 | 不得用测试数代替生命周期审计或 P1--P12 执行 |
| Extent release 修复 | completion/eviction physical release chain 已实现；154-test 联合回归通过 | tier-local release、terminal-zero 与既有路径的代码级回归成立 | 104732 请求终态聚合失败，尚无有效 post-fix integration lineage |
| 三层 JSON/runtime 编译 | P1--P12 2,134/2,134 在显式 frozen modeled bindings 下可编译 | request/failure/deployment 可确定性投影到运行时；Qwen3-32B revision gate 已绑定 | 不得称为 2,134 个 simulation cells 或论文数据 |
| P11 observed 输入 | 30 workflows、13,918 rows；manifest/request/predictor hashes 已冻结 | 可复核 workload/causal predictor 输入已建立 | 0.92857 simultaneous coverage 低于 0.95，且 0 个 nonzero safe LCB；不是 P11 结果 |
| P11 102845 smoke | small/seed0/recompute 的一个完整 57-request workflow，57/57 completed | 原子 workflow selection、真实 Scheduler replay 和 P0 v3 binding 可运行 | 无 Aegis protection/recovery、故障未命中 active request，不是 paired P11 或 predictor gate |
| P1--P12 正式执行 | **0 cells** | 只能描述冻结计划、runner 实现和 blocker | 不得出现新性能数字、曲线或 `paper_ready=true` |

因此“实现/测试通过”、“smoke 可运行”、“完整矩阵已执行”和“论文证据成立”是四个不同状态。
本报告不将测试计数、配置展开计数或 runner manifest 静态字段当作性能证据。

### 0.2 代表性 real-Scheduler smoke（非正式实验）

2026-08-29 的
[`20260829T100844Z_global_des_scheduler_smoke`](results/20260829T100844Z_global_des_scheduler_smoke/)
是当前 P1/P12 runner artifact。它只选了 P1 ordinal 0 和 P12 ordinal 0，
每个 cell 仅 1 个 request；[`suite_manifest.json`](results/20260829T100844Z_global_des_scheduler_smoke/suite_manifest.json)
明确记录 `smoke=true`、`modeled_simulation=true`、`paper_ready=false`、
`full_2146_cell_suite_executed=false`。

| Smoke cell | 实际执行内容 | 可以验证 | 不能验证 |
| --- | --- | --- | --- |
| P1 ordinal 0 | Qwen3-8B / recompute / `worker_1` / seed `2026082900`；1 request；10 s 注入、30 s 空状态 repair | 真实 Offline Scheduler、GlobalDES 事件序、hard failure/repair 物理 lifecycle 路径可运行 | request 在故障前已完成，`affected_original_requests=0`、无 protection/recovery job；不是 failure-impact 或 policy-comparison 证据 |
| P12 ordinal 0 | Qwen3-8B / 10 GB/s / 1K verification chunk / seed `2026082900`；1 analytical request | 2 个 protection physical jobs，共 301,989,888 bytes；2/2 analytical-vs-DES duration exact，最大误差 0 ns | 只是 modeled formula/internal-path smoke；无 recovery job、无真机校准，且 audit 仍标记 `scheduler_aggregate_accounting=true` / `strict_extent_accounting=false` |

这两个 cell 说明 runner 可生成 raw events、rank/GPU/node ledger、transfer/group 和 audit artifact；
它们不转换为正式 P1/P12 结果。

另一个
[`20260829T101529Z_global_des_scheduler_smoke`](results/20260829T101529Z_global_des_scheduler_smoke/)
执行了 P4 ordinal 0 的 `full` variant，共 8 个 controlled fork/join requests：8/8 完成，6 个 request
具有正 causal Slack；GlobalDES 记录 46 个物理 protection jobs、18,496,880,640 bytes，以及 32 张
物理 GPU、4 个 node-local DRAM pool 的时间序列。它证明 real Scheduler、causal ticket、逐卡/逐节点
ledger 和 transfer journal 可以在同一 GlobalDES smoke 中工作。

但该 artifact 生成于 request-completion/eviction 对物理 protection extent 的释放链修复之前：已完成
request 的 committed HBM/DRAM extent 仍可能留在账本中，形成 **extent-lifetime leak**。因此它及从中
派生的 [`plots_v3`](results/20260829T101529Z_global_des_scheduler_smoke_plots_v3/) 四张图只能用于定位
旧实现状态，不能作为 resource-conservation gate、容量结论、P4 消融或论文 performance evidence。
该 smoke 的 10 s failure 又发生在全部 8 个 request 于约 2.55 s 前结束之后，没有 active-source
recovery job；40 s repair 也不是故障影响证据。

以下四图均由同一个 pre-fix P4 smoke 派生；逐图 CSV/PDF/hash 和 `paper_ready=false` 声明见
[`plot_manifest.json`](results/20260829T101529Z_global_des_scheduler_smoke_plots_v3/plot_manifest.json)。
source suite manifest SHA-256 为
`656ed37fe22c918d386b79fb740a637a5d25f802e4e50cfa270ebf51323dcfc9`，plot manifest 文件 SHA-256 为
`0f8273d6c69517bb808967e0a13d94a84515003f3af48c4b29322371b353395f`。它们不是 P4 的四个
algorithm variants 对比图。

![Pre-fix P4 per-GPU HBM diagnostic](results/20260829T101529Z_global_des_scheduler_smoke_plots_v3/diagnostic_per_gpu_hbm.png)

*诊断 P4-D1（pre-fix / preliminary）：32 张物理 GPU 的 HBM 分账。终态 committed protection 不能被
解释为仍活跃请求的占用，因为该 lineage 缺 completion/eviction extent release chain。*

![Pre-fix P4 per-node DRAM diagnostic](results/20260829T101529Z_global_des_scheduler_smoke_plots_v3/diagnostic_per_node_dram.png)

*诊断 P4-D2（pre-fix / preliminary）：4 个 node-local DRAM pool。图中保留量同样受 extent-lifetime
leak 影响，不能用于容量 knee 或 DRAM tax 结论。*

![Pre-fix P4 KV transfer Gantt](results/20260829T101529Z_global_des_scheduler_smoke_plots_v3/diagnostic_kv_transfer_gantt.png)

*诊断 P4-D3（pre-fix / preliminary）：46 个 protection jobs 的路径/排队区间；可检查 transfer journal，
不能证明请求完成后的 extent 生命周期闭合。*

![Pre-fix P4 failure timeline](results/20260829T101529Z_global_des_scheduler_smoke_plots_v3/diagnostic_failure_timeline.png)

*诊断 P4-D4（pre-fix / preliminary）：10 s failure 与 40 s repair；8 个请求此前已结束，所以图中没有
active-request recovery 因果效应。*

#### 0.2.1 Post-fix P4 尝试为何仍不能升格

completion/eviction physical release chain 现已在代码中实现，tier-local release、stale-version isolation、
PENDING abort/independent-tier preservation 和 terminal protection-zero 等联合回归共 **154 tests PASS**。
这是代码级修复证据。

第一次 post-fix runner 尝试
[`20260829T104732Z_global_des_scheduler_smoke`](results/20260829T104732Z_global_des_scheduler_smoke/)
确实记录了 478 个 `extent_location_released` 事件，终态 Peer-HBM/DRAM 的 Aegis pending/committed
分账归零；但其 `requests.jsonl` 是 **7 completed + 1 scheduled**，而 `suite_manifest.json` 却把 cell
标为 `executed` 且 suite `complete=true`。这暴露 request terminal-state aggregation bug：当前 audit
将一个非终态 request 漏算为已完成。因此 104732 不是有效 lineage，不能用它证明 release-chain
端到端正确、替换 101529 四图或解锁 P4。必须修复聚合器并重新生成不可变 lineage。

#### 0.2.2 P11 原子完整 workflow smoke

[`20260829T102845Z_p11_atomic_small_scheduler_smoke`](results/20260829T102845Z_p11_atomic_small_scheduler_smoke/)
选择 P11 ordinal 0：`small / recompute / seed=2026082900`，完整执行 workflow
`4cb074e3cbf03d1709e2b44cbe575807813b` 的 57 个 requests。workload audit 明确记录
`atomic_workflow_replay=true`、`smoke_request_truncation_permitted=false`，`requests.jsonl` 为 57/57
completed；没有把 workflow 中间截断后计算 WCT。它引用 P0 v3 summary SHA-256
`773e39ec32ffd44d27950a471c65cb6d3826d1573fcd8f9177c68737f7374228`。suite manifest SHA-256 为
`697f88ce4bd99626b8c0246d683bc59ed1f91d49ad75d55b1e87ad9eae5eb161`，cell manifest 文件 SHA-256 为
`dc9ab53f66c1e02152d250bd6b315d33566dc15429166a8b31e04ea04da66adc`。

该 cell 是 recompute/no-backup smoke：0 protection jobs、0 recovery jobs；10 s worker failure 没有命中
active request，`strict_extent_accounting=false`。虽然用了冻结 Qwen3-8B snapshot path，它仍是
metadata/timing modeled simulation，不是 CUDA/真实硬件执行。它只证明一个 P11 分区能按完整 workflow
进入真实 Offline Scheduler；不能证明 AegisKV-vs-recompute paired effect、Algorithm 1、predictor 0.95
coverage 或 P11 60-cell 正式矩阵。

## 1. 执行摘要

历史实验套件包含 **S1--S9 加 E1，共 10 组实验、613 个实验 cells**。本次运行实际完成
613/613 cells，模拟 341,772 个 requests，产生 18,376 个 workflow/session observations，
输出 10 张 PNG、10 张 PDF 以及 10 份逐图 CSV。

历史 lineage 自身的严格审计结果为：

| 项目 | 结果 |
| --- | --- |
| 执行引擎 | `sglsim_complete_metadata_des` |
| 完成度 | 613/613 cells |
| 运行失败 | 0 |
| aggregate engine 内部语义与可复现性 | `PASS` |
| 目标 GPU/resource-level 语义 | **不满足** |
| metadata-only | `true` |
| 图像 | 10 PNG + 10 PDF |

这套历史实验是独立的 **offline aggregate metadata discrete-event simulation**。SGLSim 主 serving 路径仍然是
online、真实墙钟、HTTP 服务和异步客户端；本报告中的论文实验不是在线服务压测，也不移动
真实 KV tensor。

实验结果没有被修改成预设的理论曲线。当前证据同时包含正结果、弱结果、零结果和负结果：

- S1 中 AegisKV 相对 recompute 将 harmed-session added WCT 均值降低 52.5%；
- S4 中 causal recovery priority 显著降低最大恢复 lateness；
- S6 的高 prefix-reuse 收益较弱，其中 `>=99%` 分组置信区间跨零；
- S8 不支持 AegisKV 在所有故障规模下端到端全面占优；
- E1 的 AgentX/Weka observed-DAG 效应置信区间跨零，且安全 Slack LCB 全为零；
- S7 方差较高，15 秒与 30 秒 repair 曲线出现真实 plateau。

这些数值只能描述旧聚合模型中的观察，不应继续写入新论文结论。历史机器可读审计见
[`audit.json`](results/20260829T064719Z_paper_complete_v2/audit.json)，原始分析见
[`analysis.md`](results/20260829T064719Z_paper_complete_v2/analysis.md)。

## 2. 为什么当前图像与 Mooncake PDF 中的 SGLSim 图像不同

对照文档为
[`aegiskv-experiment-plan-preset-vs-observed.pdf`](/home/comp/zhanzhao/Mooncake/output/pdf/aegiskv-experiment-plan-preset-vs-observed.pdf)，
SHA-256 为 `c807bd087576405f0cb96f366606aa6a7e51d8c9e1f47c908735237d2ef13474`。
该 PDF 生成于 2026-08-27，其第 6--16 页的 SGLSim 部分来自
`AegisKV/doc/experiments/sglsim_wct_preliminary_v1/` 下的多个旧 preliminary lineage；当前图像来自
2026-08-29 重新建立的三层 JSON suite。两者不是同一批实验的两种画法。更进一步，本报告同时保留
历史 613-cell outcome 图、P0 v4 contract 图和 P4 pre-fix mechanism diagnostics；这三类图本身也不能
互换。

### 2.1 配置和统计口径发生了哪些变化

| 维度 | PDF 中的旧 SGLSim preliminary | 2026-08-29 legacy 613-cell suite |
| --- | --- | --- |
| 结果来源 | 多个 v3.1.3/targeted preliminary lineage 拼接 | 单一 immutable lineage `20260829T064719Z...` |
| 中心部署 | 8 nodes × 8 A100，Qwen3-8B system-DP64 | 4 nodes × 8 A100，Qwen3-8B TP1/DP32 |
| Qwen3-32B | TP1/PP4，service 按权重比解析缩放 | TP4/DP8，KV geometry 按官方 config 计算 |
| Host DRAM | 64 GiB/worker | 512 GiB/node 分配给 AegisKV |
| HBM 配置 | `Mp=40 GiB`，`alpha0=0.25`，nominal 10 GiB | 6 GiB/GPU reserve，`alpha0=0.50` |
| 带宽 | same/cross GPU、local/remote DRAM 四方向 | 20 GB/s 中央 effective cross-node，10/20/25 sensitivity |
| Failure | 稀疏 failure-rate、60 s replacement、固定 probes | policy-independent shadow 上 paired conditional fail-stop；30 s repair horizon、100/50 ms modeled failover |
| Baseline | Native Mooncake、Static FCFS、DejaVu、RAIDServe、KevlarFlow、GhostServe、AegisKV | recompute、host checkpoint、neighbor replica、AegisKV 机制适配器 |
| Scale 轴 | DP/GPU scale 是主轴 | 当前 S3 只有 model × context，没有 GPU scale |
| Seeds | S1/S2 部分 10 seeds，S3--S7 多数 3 seeds | S1--S4/S6--S8/E1 为 10，S5 为 5，S9 为 3 |
| 主要 estimand | 相对 modeled Native 的 all-workflow gain | failure-added harmed-session/WCT 和 paired AegisKV-recompute effect |
| 图像组织 | 左侧虚拟预设，右侧旧 observed | 只画当前 observed，不画目标曲线 |
| S9 | RTX4090 simulator-to-physical overlay | 纯 analytical-versus-DES 公式验证 |

### 2.2 S1--S9 编号并不是稳定实验 ID

PDF 与当前 suite 重用了 S1--S9 名称，但含义不同。例如：

- PDF 的 S1 是六架构相对 modeled Native 的 worker/node-fault WCT；当前 S1 是四个机制策略在
  `worker_1` 下相对 recompute 的 harmed-session WCT；
- PDF 的 S2 是 concurrency × failure-rate；当前 S2 是 load × `no_failure/worker_1/node_1`；
- PDF 的 S3 包含 DP scale；当前 S3 只有 Qwen3-8B/32B × context；
- PDF 的 S6 是受控 reuse sweep 与 occupancy/copy cost；当前 S6 是 TraceLab observed reuse bins；
- PDF 的 S9 是 RTX4090 holdout calibration；当前 S9 完全不使用真机数据。

因此不能问“为什么同一张 S1 图数值变了”：它们实际上不是同一个 experiment contract。旧 PDF
应该作为历史问题清单和 preliminary 诊断阅读，不是当前图像必须逼近的模板。

### 2.3 为什么不应让当前曲线变得和虚拟预设一样

PDF 左栏明确是“虚拟预设（目标问题）”，右栏才是旧观测。目标图只表达希望识别的关系，不能作为
验收阈值。当前 suite 发现的 plateau、跨零区间、AegisKV 与其他保护策略持平、S8 部分场景变差、
E1 safe Slack 为零，都必须保留。正确的修复方式是统一 workload、baseline、estimand 和坐标后重跑，
不是调整模拟值或筛掉不符合目标趋势的 seed。

### 2.4 为什么 P0/P4 资源图也不会长得像 PDF outcome 图

本报告新嵌入的 P0 五图回答事件顺序、逐 GPU/节点账本、transfer path 和 invariant 是否成立；P4 四图
只展示一个 8-request smoke 的 HBM/DRAM/transfer/failure journal。PDF 第 9--14 页主要画 WCT、failure
rate、scale、reuse、repair 等 **outcome relationship**。它们的因变量、样本、部署、故障时刻和证据等级
完全不同，所以不存在“曲线应一致”的前提。

此外，P4 四图已确认来自 pre-fix extent-lifetime-leak lineage。正确动作是保留旧 hash、使用已修复的
生命周期代码，并在 request-terminal aggregation 修复后生成新 lineage/重画诊断；不是把旧 PDF 曲线
作为资源账本的视觉验收模板。

## 3. 论文评价应回答的研究问题

纯 SGLSim 论文评价建议围绕以下 RQ 组织，而不是围绕“每条曲线是否长得像预设”：

| RQ | 问题 | 主指标 |
| --- | --- | --- |
| RQ1 | AegisKV 是否降低故障影响？ | failure-conditioned WCT、all-workflow WCT、p95/p99、paired effect |
| RQ2 | 收益在什么负载、模型、context、规模和故障域下成立？ | effect heatmap、scale/context curve、failure-domain curve |
| RQ3 | Algorithm 1/2/3 各自改变了什么？ | Slack ordering、residency/action、alpha/quota、paired WCT ablation |
| RQ4 | 正常路径为容错付出多少代价？ | no-failure WCT/TTFT/TPOT、copy traffic、HBM/DRAM occupancy |
| RQ5 | 结果对 bandwidth、repair horizon、容量和 reuse 是否稳健？ | sensitivity、phase diagram、resource efficiency |
| RQ6 | 在公开 observed multi-agent DAG 上是否仍成立？ | AgentX/Weka paired WCT、predictor coverage、DAG-size strata |
| RQ7 | 模拟器内部的 transfer/recovery 计算是否自洽？ | analytical-versus-DES error、event invariant |

### 3.1 开源模拟器选型结论

[`OPEN_SOURCE_SIMULATOR_RESEARCH.md`](OPEN_SOURCE_SIMULATOR_RESEARCH.md) 的逐项审计结论是：没有一个
开源模拟器同时提供 SGLang scheduler 语义、单一全局 DES、逐 rank/GPU HBM、node-local
DRAM、竞争 KV transfer 和 hard failure/recovery。因此当前采用混合 donor，而不引入第二套调度器：

- SGLSim + pinned SGLang 作为 scheduler/control-flow 语义权威；
- Inference Lab 作为 resource graph/path contention donor；
- BLIS 作为 transfer-station/invariant donor；
- ASTRA-Sim/SimAI 作为未来 network/collective backend donor；
- Vidur 作为 predictor calibration/trace donor。

外部 donor 可替换底层 timing/network backend，不得改变 pinned SGLang continuous-batching 控制流或
AegisKV source-owned independent-tier 语义。

## 4. 当前三层 JSON 配置

### 4.1 实验代码与配置组织

当前历史 aggregate runner 的入口为 [`AegisKV/experiments`](./)。旧实验、preliminary 结果和归档脚本保留在
`AegisKV/doc/experiments/`，不会被当前 runner 读取。

新 runner 已落盘为 [`global_des_suite_runner.py`](src/global_des_suite_runner.py)：它只调用
`OfflineSchedulerCluster` 和全局 resource DES，不复用 `AegisCompleteSimulator`。它支持 cell selection、
`--check`、smoke、resume、原子 manifest 和 fail-close readiness。当前只有第 0.2 节的 P1/P12
各 1-request、P4 8-request 以及 P11 单个完整 57-request workflow 的代表性 smoke cells，并且 manifest
禁止 paper-ready/full-suite claim；P4 101529 是 pre-fix extent-lifetime-leak artifact，post-fix 104732
又有 request-terminal aggregation bug。尚无正式 P1--P12 performance lineage。新三层合同冻结在
[`configs_v3/`](configs_v3/)：deployment 层显式包含
GPU placement、HBM 分账、节点 DRAM、copy/link 拓扑和 DES 输出合同，严格 loader/validator 为
[`global_des_config.py`](src/global_des_config.py)，request 与 runtime 编译分别见
[`global_des_workload.py`](src/global_des_workload.py) 和
[`global_des_runtime_compiler.py`](src/global_des_runtime_compiler.py)。旧 [`configs/`](configs/) 只服务
历史 aggregate lineage。

每组实验严格拆分为三个 JSON 层：

| 配置层 | 文件 | 负责内容 |
| --- | --- | --- |
| Request | `configs_v3/<group>/request.json` | trace、抽样、到达过程、context 投影、workload 轴、seed |
| Failure | `configs_v3/<group>/failure.json` | hard fail-stop 类型/故障域、确定性触发时刻、数量、repair 和 paired 合同；v3 不含 MTTF/Poisson |
| Deployment | `configs_v3/<group>/deployment.json` | 模型、TP/PP/DP、物理 GPU/DRAM、链路、策略、chunk 和部署轴 |

新套件公共默认配置位于：

- [`configs_v3/common/request.json`](configs_v3/common/request.json)
- [`configs_v3/common/failure.json`](configs_v3/common/failure.json)
- [`configs_v3/common/deployment.json`](configs_v3/common/deployment.json)
- [`configs_v3/suite.json`](configs_v3/suite.json)

每条新的不可覆盖 lineage 必须保存：

- 原始三层配置和 resolved configs；
- `current_design.md`、参数来源和审稿人配置审计；
- 实际执行代码快照及 SHA-256；
- 每个 cell 的压缩原始事件和无故障 shadow schedule；
- `summary.csv`、`observations.csv`、`timeline.csv` 和 `verification.csv`；
- 图像、逐图 CSV、分析报告、manifest 和严格审计。

### 4.2 Request 层配置

#### 4.2.1 主 workload：TraceLab v0.0.2

P1--P5、P7--P10 使用固定 SHA-256 的 TraceLab DuckDB：

| 项目 | 配置 |
| --- | --- |
| 数据集 | TraceLab `v0.0.2` |
| SHA-256 | `a7bab286bc640844560850965ccf47975cf66407154132abaab90f27ec9be744` |
| 审计规模 | 665,453 LLM rounds、8,058 provider sessions、743,819 tool calls |
| 抽样单元 | provider session，而不是单独 request/round |
| 分层 | provider × session-round-count bin：`4--16/17--64/65--256/257+` |
| 每 cell | 每个 stratum 1 个完整 session，共 8 sessions；禁止 round/session 截断 |
| 到达时间 | `timing_events.event_index_1.timestamp`，以选中记录最小时间归零 |
| Load 变换 | 只允许 `relative_arrival_ns / load_multiplier`，不重新抽样 |
| 逆序策略 | 数据集中完整 21-session/23-transition 异常集合在 seeded sampling 前排除；编译时反查，禁止 clamp/reorder |
| 主要 context | 16,384 tokens |
| output cap | 2,048 tokens |

TraceLab 的输入长度不能原样放入 Qwen3：该 artifact 的 input-token 中位数为 132,092，
因此实验使用 `rank_preserving_scale_to_target_median_then_clip`：中位 prompt 目标为所选 context 的
0.5，再做逐记录 clipping audit。`0.5`、2,048-token cap 和 sample size 都在 request JSON 中明确标为
preregistered modeled choice，而不是实测。论文中必须称为“projected、load-controlled TraceLab
replay”，不能称为原样生产 trace 或硬件 workload measurement。

P5 的 `low/high/recovery` 是三个共享同一 request universe 的独立 phase cells，倍率为
`0.5/2.0/0.75`。60 秒是 nominal control interval，不是 request cutoff；边界使用
`no_request_truncation_allow_spill`，每个 phase 独立归零。

#### 4.2.2 P6：独立 controlled-prefix artifact

P6 不冒充 TraceLab 有跨 session content identity，而使用可重生成的
[`controlled_prefix_reuse-v1.json`](data/controlled_prefix/controlled_prefix_reuse-v1.json)：

| 项目 | 冻结值 |
| --- | --- |
| Artifact SHA-256 | `26b01d51a799c9c82b786525234d7c7853a65f9317c88f12292ca237333a5afe` |
| 证据标签 | deterministic synthetic controlled；`empirical_measurement=false`；不是 TraceLab |
| Content families | 4 |
| 生成 sessions | 8/family；每 seed 选 2/family，共 8 |
| Rounds | 16/session，即 128 requests/cell |
| Token shape | 15,872 input + 512 output；可控 shared-prefix 区域 11,904 tokens |
| Reuse axis | `0/0.25/0.5/0.75/1.0` |
| Identity | SHA-256 token stream recipe；family prefix 跨选中 sessions 共享，private suffix 按 session/round/reuse 区分 |

生成器和 canonical audit 位于
[`controlled_prefix_artifact.py`](src/controlled_prefix_artifact.py)。内容 identity、reference count 和
artifact bytes 必须同时通过审计；这些参数都是用于隔离 reuse 的受控模拟选择，不是观测统计。

#### 4.2.3 P0/P11/P12 与当前 blocker

- P0 使用 12 个 complete-enumeration deterministic contract scenarios；
- P12 使用 closed-form transfer cases，始终是 modeled internal verification；
- P11 的 compiled observed-DAG 已固化在
  [`data/observed/weka_agentx_causal_v1`](data/observed/weka_agentx_causal_v1/)；runner 必须核验 manifest、
  requests、source 和 predictor hashes，不能把归档原始 JSONL 或历史 E1 aggregation 静默当作输入。

一般 Aegis cell 仍必须在 admission 时提供 causal Slack ticket，placement 不得用 future completion
静默重算。TraceLab 顺序背景使用结构上可见的 singleton zero-ticket；P4 controlled fork/join overlay
使用冻结的 admission predictor；P11 使用 workflow-disjoint、request-return-only cross-fit predictor；P12
使用 analytical ticket。P11 的 workflow-simultaneous interval coverage 为 `0.928571 < 0.95`，选中请求
safe-Slack LCB 非零数为 0，所以它只能作为 conservative preliminary input，不能宣称 predictor gate
达标或 Algorithm 1 已有有效排序信号。

### 4.3 Failure 层配置

#### 4.3.1 唯一允许的运行时故障：hard fail-stop

v3 的 Failure 层不采样 MTTF/短 trace Poisson，也没有冻结旧 aggregate suite 的
“shadow critical-tail 0.5 quantile”触发器。它只将确定性虚拟时间事件编译为
`OfflineSchedulerCluster.schedule_fail_stop`；运行时缺少 hard-failure API 时必须拒绝执行，
不能退化为仅改 metadata 的 soft failure。

hard failure 的状态转移合同是：

1. `failure_inject` 在同时刻优先于 transfer/batch completion，先 fence route 并取消命中的
   in-flight compute；
2. worker/GPU/process 失效终止目标 Scheduler worker 并清空其 volatile HBM；node 失效还清空
   该节点所有 GPU 和 node-local DRAM；
3. PENDING 的受影响 tier abort；未落在故障域的独立 committed tier 不被连带回滚；
4. recovery planner 只能根据故障后仍存活的 Peer-HBM/NODE-DRAM coverage 创建完整物理
   recovery job，coverage 不足时 fail-close 或显式 recompute；
5. repair 只启动空 worker/空 volatile ledger，不复活已丢失 KV。

这一路径已有 code-level integration tests，但尚没有新的 immutable P1--P12 execution
lineage，因此不提供 WCT 或 recovery-latency 结果。

#### 4.3.2 公共冻结参数与运行时 binding

| 参数 | 默认值 | 解释 |
| --- | ---: | --- |
| 故障类型 | `fail_stop` | 不覆盖 fail-slow、网络分区或 silent corruption |
| 支持 scope | worker / GPU / process / node | GPU 在 TP>1 时仍有粒度 blocker，不得将一个 DP 当单 TP shard 清空 |
| 中央场景 | `worker_1` | 组内 failure axis 可显式覆盖 |
| 触发点 | `10,000,000,000 ns` | deterministic global virtual time，同时 tie 使用 GlobalDES priority |
| short-run Poisson | `false` | v3 不生成 hazard/MTTF 事件 |
| repair horizon | 30 s | modeled capacity-restoration horizon，不是硬件维修分布 |
| replacement delay | 100 ms | modeled、未同拓扑校准 |
| queue/routing outage | 50 ms | modeled、未同拓扑校准 |
| target selector | `dense_first_v1` 或 `seed_mod_dense_order_v1` | 必须在 run-owned binding 显式选择，不允许猜测 |
| repair-delay composition | `serial_sum_v1` 或 `overlap_max_v1` | 显式规定 100/50 ms 是相加还是重叠；缺失则 fail-close |
| Paired 合同 | true | 同 seed、request universe、failure target/time 并要求 shadow-schedule hash；不代表已生成 paired 数据 |

各组仅冻结自己所需的差异：P1 为 `worker_1/node_1`；P2 加 `no_failure`；P3 用
`constant_pre_injection_gpu_exposure_v1` 依据已编译物理 GPU 数缩放 10 s 参考触发时刻；
P7 为 1/5/15/30 s × worker/node；P8 为 1/2/4 workers 与 1 node。P5、P9、P12 显式禁用
failure。v3 没有 MTTF 或 hazard-weighted axis；如未来需要 field-incidence 分析，必须作为新的
、有独立来源和估计量的配置层变更，不能把历史 9,600 GPU-hour proxy 写回本计划。

#### 4.3.3 Causal ticket 是保护/恢复的准入条件

当前请求编译器已经为各 workload 明示 ticket 来源，但 ticket 存在不等于其预测有效：

- TraceLab 不提供 observed parallel residual DAG；其顺序背景只构造 admission-visible singleton frontier，
  精确 zero Slack，禁止伪造正 Slack；
- P4 叠加 8 个 modeled four-branch fork/join workflows；正 Slack 只来自冻结 admission predictor，绝不读取
  simulated completion。该 workload 用于机制 smoke，不冒充 TraceLab observed DAG；
- P11 ticket 只使用 request-return prefix 可见特征和 workflow-disjoint 5-fold cross-fit；未建模 request
  显式 `usable=false, slack=0`，不能关闭 Aegis protection；
- P11 当前 95% workflow-simultaneous coverage 只有 0.928571，未过预注册 0.95 门槛，且 selected
  nonzero safe LCB 为 0；
- P12 只能使用显式 analytical ticket，其输出永远是 modeled/preliminary/non-paper-ready；
- 任何路径仍不得用完成后 request duration、WCT 或 recovery outcome 倒推 Slack。

### 4.4 Deployment 层配置

#### 4.4.1 集群与内存

| 项目 | 配置 |
| --- | --- |
| 节点数 | 4 |
| GPU/节点 | 8 × NVIDIA A100-40GB |
| 总 GPU | 32 |
| Host memory/节点 | 1,024 GiB |
| HBM runtime / primary / radix 预算 | 6 / 8 / 1 GiB per GPU |
| Aegis Peer-HBM 中央预算 | 6 GiB/GPU |
| DRAM runtime / HiCache 预算 | 128 / 256 GiB per node |
| Aegis node-DRAM 中央 quota | 512 GiB/node，按节点独立，禁止合并为 global pool |

中央 copy-resource 输入为显式的 modeled reservation calendar：

| 资源 | 带宽 | setup | 竞争语义 |
| --- | ---: | ---: | --- |
| NVLink/NVSwitch | 100 GB/s | 1 µs | exclusive reservation calendar |
| PCIe | 25 GB/s | 2 µs | exclusive reservation calendar |
| NIC/RDMA | 25 GB/s | 3 µs | exclusive reservation calendar |
| cross-node fabric | 25 GB/s peak，20 GB/s central effective | 5 µs | exclusive reservation calendar |
| node DRAM read/write lane | 40 GB/s | 1 µs | 每节点独立 read/write lane |

P7/P12 显式 sweep 跨节点 effective bandwidth `10/20/25 GB/s`。这些值都是冻结的
modeled input，不是 A100 端到端拷贝实测。路径时间是所需资源可用时刻的最大值，加配置
setup 和 `bytes / bottleneck_bandwidth`；不使用旧报告的 270/22.5 GB/s diagnostic 或
0.10/0.15 ms aggregate setup。

#### 4.4.2 模型与 KV geometry

| 模型 | 部署 | Context cap | KV bytes/token | HBM protection reserve |
| --- | --- | ---: | ---: | ---: |
| Qwen3-8B BF16 | TP1 / DP32 | 40,960 | 147,456 | 6 GiB/replica |
| Qwen3-32B BF16 | TP4 / DP8 | 40,960 | 262,144 | 24 GiB/TP4 replica，即 6 GiB/GPU |

KV footprint 由官方模型 geometry 直接计算：

- Qwen3-8B：`2(K/V) × 36 layers × 8 KV heads × 128 × 2 BF16 bytes = 147,456 bytes/token`；
- Qwen3-32B：`2 × 64 × 8 × 128 × 2 = 262,144 bytes/token`。

Qwen3-32B BF16 权重约 65.6 GB，无法 TP1 放入 A100-40GB，因此使用 TP4。当前三层配置已经冻结
Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218` 和 Qwen3-32B revision
`9216db5781bf21249d130ec9da846c4624c16137`，此前的 32B revision gate 已解除。
不过，model ID/revision/geometry 仍不等于完整运行 lineage：正式 run 还必须绑定 simulator
implementation hash，以及 timing source identity + SHA-256。timing 只能
标为 `calibrated` 或显式授权的 `modeled_not_calibrated`；本 v3 deployment 不包含旧 aggregate
runner 的 4/9 ms setup 和逐 token 线性系数，因此本报告也不再将那些数值称为当前配置。

#### 4.4.3 Chunk 与三算法控制参数

| 参数 | 数值 |
| --- | ---: |
| 默认 prefix chunk | 4,096 tokens |
| 默认 decode chunk | 256 tokens |
| P12 verification chunk axis | 1,024 / 4,096 / 8,192 tokens，仅 analytical 验证 |
| AIMD control window | 1,000 ms |
| 初始 alpha | 0.50 |
| alpha 范围 | 0.10--1.00 |
| Multiplicative decrease | 0.50 |
| Additive increase | 0.05 |

P0--P11 的 current-design 默认 chunk 为 4,096/256；P12 的 `1K/4K/8K` 是 analytical
verification axis，不是将运行时默认静默改为 1K。冻结 JSON 没有 `stable windows=3`
字段，不应从旧 aggregate 实现把它复制到当前计划。

#### 4.4.4 策略

| 策略 | Protection | Recovery ordering | AIMD | 角色 |
| --- | --- | --- | --- | --- |
| `recompute` | 无备份 | FIFO | 否 | recompute baseline |
| `host_checkpoint` | DRAM only | FIFO | 否 | host-checkpoint mechanism adapter |
| `neighbor_replica` | Peer-HBM only | FIFO | 否 | neighbor-copy mechanism adapter |
| `aegiskv` + `full` | independent HBM + DRAM | causal ticket 决定 | 是 | 完整 current design；ticket 缺失则 fail-close |
| `aegiskv` + `no_algorithm_1` | independent HBM + DRAM | `aegis_l1_l2.disable_slack_ordering=true` | 是 | Algorithm 1 runtime adapter 已显式绑定 |
| `aegiskv` + `static_placement` | independent HBM + DRAM | 容量匹配的静态 placement | 是 | Algorithm 2 ablation |
| `aegiskv` + `no_algorithm_3` | independent HBM + DRAM | causal ticket 决定 | 否，alpha 固定 | Algorithm 3 ablation |

`host_checkpoint` 和 `neighbor_replica` 是机制适配器，不冒充 FailSafe、GhostServe 或 DejaVu 的完整实现。
严格 compiler 还将 P10 DRAM budget 绑定为具体 **per-node** quota，并按 P3 的已编译物理 GPU 数应用
`10 s × reference_gpu_count / selected_gpu_count`。在显式 frozen modeled bindings 下 P1--P12 的
2,134/2,134 cells 均可编译；上表和这个计数仍只是运行时投影合同，不表示任何 variant 已执行。

## 5. 正式论文建议执行的纯 SGLSim 实验

当前 613-cell suite 只是一条历史 aggregate lineage，不满足本文档的资源粒度。正式论文矩阵命名为
P0--P12，避免继续与历史 PDF/S1--S9 的编号冲突。下表 cell 数已经由冻结的 v3 JSON 和 validator
机械展开；suite 仍明确标记为 `planned_unexecuted`。任何尚未执行的 cell 都不得用旧结果或预设曲线代替。

### 5.1 建议矩阵

| 组 | RQ | 固定项与 sweep | Seeds | Cells | 论文图像 |
| --- | --- | --- | ---: | ---: | --- |
| P0 Simulator contract | RQ7 | 并发/TP-PP shard/transfer path/failure window/tie 共 12 个确定性场景 | 1 | 12 | invariant table + event-order/resource conservation |
| P1 Overall | RQ1 | 4 policies × 2 fault domains × 2 models | 10 | 160 | Absolute WCT ECDF + paired-effect forest plot |
| P2 Load/fault | RQ1/RQ2 | 4 loads × 3 failure scenarios × 2 policies | 10 | 240 | Load × failure heatmap，mean/p99 两行 |
| P3 Scale/context | RQ2 | 4 node counts × 4 contexts × 2 models × 2 policies | 10 | 640 | Model 分行、context 分面、x=nodes 的 effect curves |
| P4 Algorithms | RQ3 | Full / no Alg.1 / static placement / no Alg.3 | 10 | 40 | 机制三 panel + paired WCT forest |
| P5 AIMD | RQ3/RQ5 | 3 个独立 load-phase cells × 4 HBM budgets | 5 | 60 | alpha/quota/used/pressure phase small-multiples |
| P6 Reuse | RQ5 | 5 reuse levels × 2 replica placements × 2 policies | 10 | 200 | Copy cost、residency、recovery hit、WCT 四 panel |
| P7 Bandwidth/repair | RQ5 | 3 bandwidths × 4 repair horizons × 2 fault domains | 10 | 240 | Faceted bandwidth-repair response surface |
| P8 Failure severity | RQ2 | 1/2/4 workers + node × 2 models × 2 policies | 10 | 160 | Absolute impact + paired reduction by domain |
| P9 Healthy overhead | RQ4 | 4 loads × 2 models × no-failure AegisKV/control | 10 | 160 | TTFT/TPOT/WCT overhead forest + copy traffic |
| P10 Capacity | RQ5 | 4 HBM budgets × 3 DRAM budgets × 2 models | 5 | 120 | Benefit/coverage/eviction phase diagrams |
| P11 Observed DAG | RQ6 | 3 AgentX DAG-size strata × 2 policies | 10 | 60 | Paired WCT + predictor coverage/Slack distribution |
| P12 Internal check | RQ7 | 2 models × 3 bandwidths × 3 verification chunks | 3 | 54 | analytical vs DES parity + error distribution |
| **冻结计划总计** |  |  |  | **2,146** | **13 组图** |

### 5.2 每组精确配置和画法

P0--P12 继承三层配置形式与 source-owned independent HBM/DRAM 语义，但使用新的 v3 schema。除非某组在下文
显式 sweep，公共固定项为：4 nodes × 8 A100-40GB、Qwen3-8B TP1/DP32、TraceLab projected
16K context、load=1.0、`worker_1` deterministic hard fail-stop（10 s virtual time）、30 s repair horizon、100 ms replacement、
50 ms routing outage、20 GB/s effective cross-node bandwidth、6 GiB/GPU HBM reserve、
512 GiB/node AegisKV DRAM、4,096-token prefix chunk 和 256-token decode chunk。10-seed 组固定使用
`2026082900`--`2026082909`，5-seed 组使用前 5 个，3-seed 组使用前 3 个。正式执行前，每组都必须
分别固化为 `request.json`、`failure.json` 和 `deployment.json`，并由 validator 从三层笛卡尔积计算
cell 数。P5/P9/P12 在自己的 failure JSON 中显式禁用故障；其他组如有轴覆盖则以该组为准。
正式 run 还必须冻结 model/simulator/timing、target selector 和 repair-delay composition 的 run-owned
bindings。本节中的 P 组是待执行实验合同，不是当前 613-cell 结果的别名。

#### P0：全局时钟与资源守恒合同

- 两个独立 DP 同时开始相同 service，验证 makespan 为 `max` 而不是串行求和；
- TP1/TP4/PP2 场景验证 KV bytes 按 layer/KV-head shard 落到每张物理 GPU；
- same-node P2P、cross-node P2P、source→DRAM、DRAM→recovery GPU 四条路径验证资源 reservation；
- failure 分别落在 PENDING 窗口、COMMITTED 后、target-first、source-first 和 node failure；
- 同时间事件验证固定 priority/sequence，重复运行验证 event hash；
- 每个场景必须通过 HBM/DRAM capacity、state transition、shard ACK、extent conservation 和
  no-HBM-to-DRAM-edge invariant。

P0 只回答 GlobalDES 内核合同是否自洽，不产生性能优越性结论。P0 任一 cell 失败时，P1--P12
全部 fail-close。

##### P0 当前合同状态与权威 lineage

2026-08-29 已执行并固化
[`20260829T112907Z_global_des_p0_contract_v4/P0/summary.json`](results/20260829T112907Z_global_des_p0_contract_v4/P0/summary.json)。
该 lineage 的 **GlobalDES contract gate 12/12 cells PASS**，每个 cell 的 10 项 invariant 均通过；这说明
单一全局虚拟事件序、确定性同时间排序、显式 TP/PP shard、物理 GPU/节点账本、迁移资源路径、
PENDING/COMMITTED/ABORTED 状态和合同中的故障窗口在这些确定性内核场景下自洽。它不等价于完整实验
执行链已完成，权威 summary 明确给出：

| P0 claim | 权威 v4 值 | 可作出的结论 |
| --- | --- | --- |
| `configured_p0_contract_gate_passed` | `true` | 12 个内核合同场景通过 |
| `controller_bridge_target_rotation_closed_loop` | `true` | `target_first_and_node_failure` 实际调用 current controller + bridge rotation 闭环 |
| `full_scheduler_integration` | `false` | 不能声称真实 SGLang-aligned Scheduler/Aegis 端到端链已通过 |
| `p1_p12_execution_ready` | `false` | P1--P12 继续 fail-close，不得启动正式结果解释 |
| `performance_result` | `false` | 下列图不是性能收益、系统加速或方案优越性证据 |
| `real_hardware_measurement` | `false` | 全部时间和资源轨迹均为离线 modeled evidence |

ordinal 10 `target_first_and_node_failure` 的 4/4 场景断言具体验证：

- 目标 GPU 先失效时，`route_version=7` 不变，旧 target HBM owner 显式 release，不重标签为 replacement owner；
- replacement 只以 HBM (`allowed_tiers=["l1"]`) 重准入并 commit，source-owned DRAM group/id/
  bytes 原样保留（旋转前后均为 603,979,776 bytes）；
- 随后 source node 失效时，仅从健康 replacement HBM 恢复到 retry GPU；
- replacement node 再失效后，stale/unhealthy tier 被拒绝，新恢复计划 fail-close 为
  `selected_tier="recompute"` 且无 transfer job。

该 cell 的 10/10 invariant 通过，`kv_extent_conservation.audit_errors=[]`、
`strict_extent_accounting=true`，bridge invariant audit 也为空。这些断言只属于该 contract cell，
不会改写上表的负向 readiness claim。

较早的 v1/v2 lineage
[`20260829T091507Z_global_des_p0_contract_v1/P0`](results/20260829T091507Z_global_des_p0_contract_v1/P0/)
使用了容易被误读为“后续实验已可执行”的旧字段 `p1_p12_unblocked=true`。该 v1 lineage **保留且不可删除**，
用于审计生成历史，但其 claim 措辞已被后续 lineage superseded。v3 亦保留为不可变历史输出；
此后当前状态只引用 v4 的 `p1_p12_execution_ready=false`。

下列五张图来自同一 v4 lineage；逐图 CSV、PDF、SHA-256 和源 cell lineage 见
[`figures/manifest.json`](results/20260829T112907Z_global_des_p0_contract_v4/P0/figures/manifest.json)。
`summary.json` SHA-256 为
`16b29d16e9718387e96d6b61f402584259de3a618c2a14ab715a4907bc1f027d`，figure manifest SHA-256 为
`0fd988ca1cb2b6feb8026ba42a26b7c0feb0c3aae4f4020cbd5a0eafa22b2b11`。

![P0 global event order trace](results/20260829T112907Z_global_des_p0_contract_v4/P0/figures/event_order_trace.png)

*图 P0-A（modeled / contract-only / non-performance）：代表性全局事件顺序与时间轨迹，左侧展示 TP4
迁移事件，右侧展示同一时间戳内按 priority、sequence 稳定派发的见证。*

![P0 per-GPU HBM accounts](results/20260829T112907Z_global_des_p0_contract_v4/P0/figures/gpu_hbm_accounts.png)

*图 P0-B（modeled / contract-only / non-performance）：逐物理 GPU 的 HBM 分账轨迹，显示 weights、
runtime、primary/prefix KV 以及 Aegis pending 到 commit 或 abort 的合同状态变化；
target-first 行将旧 target DP1 与 replacement DP2 分开，显示旧 owner release 和 HBM-only 重新 commit。*

![P0 per-node DRAM accounts](results/20260829T112907Z_global_des_p0_contract_v4/P0/figures/node_dram_accounts.png)

*图 P0-C（modeled / contract-only / non-performance）：逐节点 DRAM 分账与节点故障轨迹；它验证节点本地
账本和故障域记账；target 旋转期间 source node0 的 DRAM owner/id/bytes 保持不变，
直到 source-node failure 才归零。它不代表真实 DRAM 带宽或硬件容量测量。*

![P0 KV transfer Gantt](results/20260829T112907Z_global_des_p0_contract_v4/P0/figures/kv_transfer_gantt.png)

*图 P0-D（modeled / contract-only / non-performance）：KV 迁移 Gantt 与资源路径见证，覆盖同节点 Peer-HBM、
跨节点 Peer-HBM、source→DRAM、DRAM→recovery GPU 以及 PENDING 窗口 abort。*

![P0 invariant matrix](results/20260829T112907Z_global_des_p0_contract_v4/P0/figures/invariant_matrix.png)

*图 P0-E（modeled / contract-only / non-performance）：12 个确定性合同场景 × 10 项 invariant 的 PASS
矩阵；绿色单元仅表示模拟器内核合同成立，不表示 full Scheduler integration 或性能优越性。*

#### P1：总体故障恢复效果

- Workload：同一批 TraceLab provider sessions，16K 中央 context，load=1.0；
- Policies：`recompute`、`host_checkpoint`、`neighbor_replica`、`aegiskv`；
- Faults：`worker_1` 与 `node_1`，Qwen3-8B 和 Qwen3-32B 分开；
- 必须使用同 seed、同 request universe、同 shadow failure target；
- 左 panel 画 absolute harmed-workflow WCT ECDF，不平移曲线；
- 右 panel 画每个 model/fault/policy 相对 recompute 的 paired mean/p95 effect 与 95% CI；
- 不把机制适配器写成相关论文的 exact reproduction。

这张图回答“是否有效”，应作为正文第一张 evaluation figure。历史 S1 只覆盖 Qwen3-8B +
`worker_1`，因此不是 P1。TraceLab 背景现在会显式给出 admission-visible zero-ticket，不再因缺字段
阻塞保护；但它没有 observed parallel residual DAG，不能为 Algorithm 1 产生正 Slack。P1 paired result
仍须等待第 11 节的生命周期/failure/replay 正确性门和正式执行。

#### P2：负载与故障条件

- Load multiplier：`0.5/1.0/1.5/2.0`；
- Failure scenario：`no_failure/worker_1/node_1`；
- Policies：recompute 与 AegisKV；Qwen3-8B 为主；
- 主 heatmap 画 paired failure-added WCT，不能把无故障 overhead 与 conditional failure harm 混合；
- 下方小 panel 同时画 final alpha、pressure-target fraction 和 Peer-HBM occupancy，解释高负载退化；
- 本冻结组不画 hazard-weighted secondary axis；它没有 field MTTF 输入，不得从历史 proxy 补数。

历史 S2 缺少 load=2.0 和机制资源 panel，其 180 cells 不计入 P2 执行数。

#### P3：模型、context 与部署规模

- Models：Qwen3-8B TP1 与 Qwen3-32B TP4；
- Node count：`1/2/4/8`，每节点 8 GPUs，DP 按模型 topology 派生；
- Context：`8K/16K/32K/40K`；
- Policies：recompute/AegisKV paired；
- 为保持 pre-injection per-GPU exposure，触发时刻按
  `10 s × reference_gpu_count / selected_gpu_count` 调整，参考是 4 nodes × 8 GPUs，按 half-up 到 ns；
- 每个 scale cell 必须保持 per-GPU arrival 和 per-GPU fault exposure 不变；不能固定 12 probes 后把
  affected fraction 被 DP 分母稀释的趋势称为 scale efficiency；
- 图中 8B/32B 分行，absolute WCT 与 paired reduction 同时展示，不跨模型做错误排名。

历史 S3 只有 model × context，没有 node-count、per-GPU fault exposure 和 recompute pair；这是相对
旧 PDF 最明显的实验缺口。

#### P4：Algorithm 1/2/3 消融

四个 preregistered variants 应是：

1. Full AegisKV；
2. no Algorithm 1：同 placement/control，但 Slack ticket 关闭、recovery FIFO；
3. no Algorithm 2：使用固定、容量匹配的 static placement，不运行 SlackLandlord；
4. no Algorithm 3：固定 alpha，不运行 AIMD。

图必须同时显示：Algorithm 1 的 ordering/lateness、Algorithm 2 的四种 residency/action、Algorithm 3
的 alpha/quota response，以及 seed-level paired WCT effect。只画 WCT 柱状图会把 ppm/弱机制效应
隐藏在总 WCT 中。冻结 P4 已用
`algorithm_variant=full/no_algorithm_1/static_placement/no_algorithm_3` 表达这四个 variant，
P4 controlled fork/join causal overlay 与 `no_algorithm_1` 的
`aegis_l1_l2.disable_slack_ordering=true` runtime adapter 均已接通。2,134-cell compiler audit 中 P4
40/40 可编译，但正式四 variant paired execution 仍为 0；101529 smoke 只跑了 `full` 的 8 requests，
且受 pre-fix extent-lifetime leak 影响，不能生成消融数字。

#### P5：AIMD 控制闭环

- `low/high/recovery` 是共享同一 request universe 的三个独立 phase cells，倍率为
  `0.5/2.0/0.75`；
- 每个 phase 独立以 0 为起点；60 s 只是 nominal control interval，不截断 request，边界允许 spill；
- HBM protection budget：`2/4/6/8 GiB per GPU`；
- 对三个独立 phase 画相同横坐标的 small-multiple：pressure、alpha、quota bytes、used bytes；
- 另画累计 reclaim、HBM_ONLY direct reclaim 和 admission stop/resume marker；
- 因为 alpha/state 不跨 cell 传递，不得把三张独立图拼接成“同一控制器 low→high→recovery”因果轨迹。

历史 S5 只可作为旧 aggregate 控制律证据。如论文必须回答“同一 controller 在负载跳变后如何恢复”，
需在执行前将 P5 冻结为单 cell 内有状态连续的 phase replay，并重新展开/审核 cell 数；
不能运行后拼接。

#### P6：Shared-prefix reuse

- Controlled reuse：`0/0.25/0.5/0.75/1.0`；
- Placement：same-node Peer-HBM 与 cross-node Peer-HBM 分开；
- Policies：recompute/AegisKV；
- 同时报告 protection bytes/request、HBM/DRAM occupancy byte-ms、eviction/reclaim、recovery-token hit、WCT；
- content identity、per-tier epoch、PENDING/COMMITTED 去重和 refcount 必须审计；
- copy ratio 下降与 DRAM occupancy 上升可以同时成立，不能只挑一个指标解释。

历史 S6 是 TraceLab observed reuse bin，不是受控 0--1 reuse sweep；它可作为外部补充，不替代 P6。

#### P7：带宽与 repair horizon

- Effective cross-node bandwidth：`10/20/25 GB/s`；
- Repair horizon：`1/5/15/30 s`；
- Fault domain：worker 与 node 分面；Qwen3-32B 为 stress model；
- 横轴和颜色不能同时编码同一个变量；建议每个 fault domain 一个 panel，x=bandwidth，line=repair；
- 主分母是 all-workflow/session，failure-target conditional effect 作为 secondary panel；
- 同时标出 replacement route 是否在 repair 前激活，避免把 plateau 误读为绘图错误。

历史 S7 只有 node failure，其 3×4×10 bandwidth/repair 轴不计入 P7 执行数。

#### P8：故障数量与故障域

- Failures：`worker_1/worker_2/worker_4/node_1`；
- Models：Qwen3-8B/32B；Policies：recompute/AegisKV；
- 第一 panel 画 absolute all-workflow p95 added WCT；
- 第二 panel 画 paired AegisKV-recompute effect 与 CI；
- 第三 panel 画 recovery lateness/action mix，解释 WCT 与恢复优先级为何可能方向不同；
- worker count 和 node domain 必须分开，不能把同节点副本在 node fault 中仍视为可用。

历史 S8 已覆盖 Qwen3-32B 的 80 cells，但这些不是 P8 GlobalDES cells。

#### P9：正常路径开销

- `no_failure` only；load=`0.5/1.0/1.5/2.0`；Qwen3-8B/32B；AegisKV 与 no-protection control；
- 指标：mean/p99 WCT、TTFT、TPOT、throughput、protection bytes/s、copy-queue stall、HBM/DRAM byte-ms；
- 图采用 paired percentage overhead forest plot，并单独画绝对 throughput；
- 不能从 failure run 中反推正常路径 overhead。

当前 GlobalDES suite 没有独立、完整的 healthy-path overhead result；这是论文必须新增的主实验。

#### P10：HBM/DRAM 容量敏感性

- HBM reserve：`2/4/6/8 GiB/GPU`；AegisKV DRAM：`128/256/512 GiB/node`；
- Qwen3-8B/32B 分面；固定中央 load/context/fault；
- 三幅 phase diagram：failure-added WCT、protected-prefix coverage、eviction/recompute fraction；
- 标出 capacity knee，而不是选一个“最好预算”后只报告该点。

旧 aggregate suite 固定 6 GiB/GPU 和 512 GiB/node，没有容量 axis；旧 PDF 的 alpha-max sweep 也不能替代
当前设计下独立 HBM/DRAM admission 的二维容量实验。

#### P11：公开 observed-DAG 外部有效性

- 保留 AgentX/Weka 完整 DAG、workflow 内时序和 prefix hash；
- 按 requests/workflow 的冻结等基数秩三分位分成 small/medium/long 三个 preregistered strata；
- 每 seed 每 strata 选择相同数量 workflow，跨 seed 无放回或明确报告重叠；
- 图 1 画 paired WCT effect；图 2 画 predictor interval coverage、safe-Slack nonzero rate 和 action mix；
- safe Slack 为零时必须将收益归因于 protection extent，而不是 Algorithm 1。

实际输入已编译为
[`weka_agentx_causal_v1`](data/observed/weka_agentx_causal_v1/)：源 393 workflows/98,827 requests 中
126 个 workflow 含 observed parallel stage；按实际 requests/workflow 排序后 small/medium/long 各 42
个 eligible workflows，边界分别为 `30--162`、`163--470`、`475--4114`。10 个 seed × 3 strata
各无放回选择 1 个 workflow，得到 30 workflows/13,918 rows。manifest SHA-256 为
`9e6ca27ebc70107836bc39f8d5ed945de9af5e2224a3ab6cfbaa21d1926f0eeb`，requests SHA-256 为
`7f0a9b18aa30dbf03dd349e15fc9f10f9d57f9acb041192d5a55404a0248551b`，request-universe hash 为
`587278b60852e84fd258172ae9c9458d498f1316e502e68820b2bea80a54b9ae`，predictor-lineage hash 为
`0e6c9acf78c644ee554ec8dcc9f15f07b5e07c97b79be31997eda37a01039dc3`。1,847,151,435-byte source
JSONL SHA-256 为 `29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276`。

6,047/13,918 requests（43.45%）有 modeled admission ticket，7,871 显式 fail-close-zero；safe-Slack
LCB 非零数为 0。request-level remaining interval coverage 是 0.98175，但预注册的关键
workflow-simultaneous coverage 只有 `0.928571 < 0.95`，因此 predictor gate 未通过。该产物是真实
observed workload 的 **preliminary modeled input**，不是硬件校准或 P11 execution result。历史 E1
只覆盖不超过 128 requests 的 20 个 workflow，仍不能升格为 P11。

#### P12：模拟器内部一致性

- Models：8B/32B；bandwidth=`10/20/25 GB/s`；verification prefix chunks=`1K/4K/8K`；
- 画 analytical recovery time 与 DES event time 的 y=x scatter；
- 下方画 absolute/relative error distribution，并分别标记 DRAM、Peer-HBM、recompute；
- 这是 simulator verification/appendix figure，不是性能收益图，也不使用任何真机数据。

历史 S9 是 18-cell 的 1K-chunk aggregate analytical verification，不是 P12。当前 runner
可为 P12 生成明示 analytical ticket，但任何 smoke 结果都必须保持
`modeled/preliminary/non-paper-ready`。

### 5.3 正文与附录的图像安排

| 位置 | 建议图 |
| --- | --- |
| Simulator appendix | P0 invariant/conservation 与 representative global event trace |
| 正文主图 1 | P1 overall absolute WCT + paired effect |
| 正文主图 2 | P2 load/failure heatmap + pressure explanation |
| 正文主图 3 | P3 model/context/scale small multiples |
| 正文主图 4 | P4 three-algorithm mechanism + WCT ablation |
| 正文主图 5 | P5 三个独立 phase small-multiples + P9 healthy overhead（可组成一张双栏图） |
| 正文主图 6 | P11 AgentX/Weka external validity |
| 正文或附录 | P7/P8 failure sensitivity，视页数选择 |
| 附录 | P6 reuse、P10 capacity、P12 simulator verification、完整 seed-level tables |

每篇资源机制评价还必须包含至少一张正文或附录的 **per-GPU stacked HBM timeline**、一张
**per-node DRAM timeline**、一张 **KV transfer Gantt** 和一次 **failure zoom-in**。这些图由 P5/P8/P9
的代表性预注册 seed 生成；旧 aggregate `timeline.csv` 不能用作替代。

### 5.4 统一绘图规则

所有正式图遵循以下约束：

- 同时给 absolute metric 和 paired delta，避免绝对量程掩盖小效应；
- CI 的独立单位是 seed/workflow partition，不把同 seed 内 requests 当独立样本；
- `no_failure` overhead 与 conditional failure harm 分开；本 v3 没有 hazard 输入，不画 hazard-weighted risk；
- 同一图的 policies 必须使用相同 request、failure target 和 shadow schedule；
- 不画虚拟目标曲线，不要求单调，不筛除负值、plateau 或跨零 CI；
- 每张图的 caption 写清 workload、model/TP/DP、fault domain、seed 数、estimand 和 simulation-only；
- 每张图必须有独立 CSV，并能从 immutable lineage 重绘；
- 每张资源图还必须能追溯到 `events`、`gpu_hbm_timeseries`、`node_dram_timeseries`、`kv_extents` 和
  `transfers` 原始表；
- baseline 是机制适配器时写 adapter，不把它标成相关工作的 exact implementation。

## 6. 历史已执行的 613-cell 聚合矩阵（降级证据）

本节及第 8 节只保留旧 aggregate lineage 的可复现历史，不计入新的 P0--P12 论文结果。当前正式
P1--P12 执行数仍为 **0 cells**；P0 的 12 cells 仅属于 GlobalDES 内核合同验证。

| 组 | 研究问题 | Workload | 主要轴 | Seeds | Cells | 图 |
| --- | --- | --- | --- | ---: | ---: | --- |
| S1 | 策略对比 | TraceLab | 4 policies × 1 worker failure | 10 | 40 | policy WCT CDF |
| S2 | 负载与故障域 | TraceLab | 3 loads × 3 failures × 2 policies | 10 | 180 | heatmap/small multiples |
| S3 | 模型与 context | TraceLab | 2 models × 4 contexts | 10 | 80 | model-context lines |
| S4 | 三算法消融 | controlled fork/join | 4 policies × 3-worker ordering stress | 10 | 40 | ablation bars |
| S5 | AIMD 动态 | controlled fork/join | 3 loads | 5 | 15 | alpha timeline |
| S6 | Prefix reuse | TraceLab | 2 policies，按 reuse 分层 | 10 | 20 | reuse-stratified effect |
| S7 | 带宽与 repair | TraceLab | 3 bandwidths × 4 repair horizons | 10 | 120 | sensitivity lines |
| S8 | 故障数量/域 | TraceLab | 4 failures × 2 policies | 10 | 80 | failure-domain bars |
| S9 | Transfer model | controlled fork/join | 2 models × 3 bandwidths | 3 | 18 | analytical verification |
| E1 | 公开 observed DAG 外部有效性 | AgentX/Weka | 2 policies | 10 | 20 | paired external validity |
| **总计** |  |  |  |  | **613** | **10 组图** |

当前结果与建议论文矩阵的对应关系如下。这里的“部分覆盖”表示已有可复用证据，但不能通过改名或
重画直接视为该组正式论文实验已经完成。

| 当前组 | 对应建议组 | 覆盖状态 | 仍需补充的主要轴 |
| --- | --- | --- | --- |
| S1 | P1 | 部分覆盖 | Qwen3-32B、node failure、完整 paired effect panels |
| S2 | P2 | 部分覆盖 | load=2.0、资源压力解释 panel |
| S3 | P3 | 部分覆盖 | 1/2/4/8 nodes、recompute pair、per-GPU exposure control |
| S4 | P4 | 部分覆盖 | capacity-matched static placement，即真正 no Algorithm 2 |
| S5 | P5 | 部分覆盖 | low-high-recovery phase、HBM budget axis |
| S6 | P6 | 补充证据 | controlled reuse level 和 replica-placement sweep |
| S7 | P7 | 部分覆盖 | worker fault panel、all-workflow/conditional 双口径 |
| S8 | P8 | 部分覆盖 | Qwen3-8B、recovery action mix |
| S9 | P12 | 部分覆盖 | 4K/8K verification chunks |
| E1 | P11 | 部分覆盖 | medium/long DAG strata、predictor coverage panel |
| 无 | P9 | 未覆盖 | no-failure healthy-path overhead |
| 无 | P10 | 未覆盖 | HBM × DRAM capacity sweep |

## 7. 指标与统计方法

主要指标包括：

- workflow/session completion time；
- failure-added WCT 的 mean、p50、p95；
- harmed-workflow/harmed-session conditional added WCT；
- request latency p50/p95/p99；
- recovery deadline lateness；
- recovery action、恢复 token 数和 transfer duration；
- alpha/HBM quota trajectory；
- analytical-versus-DES transfer error。

不确定性使用独立 deterministic sampling seed 作为统计单元，报告双侧 95% Student-t confidence
interval。同一 seed 内的 workflow/session 用于 paired comparison，不被错误地当成独立重复样本。

S1--S3 使用 harmed-session conditional 指标，是因为一次强制 critical request failure 不会改变大量
unaffected sessions 组成的总体 p95。S7/S8 则报告 all-session p95，用于展示大范围故障的系统影响。

## 8. 历史 613-cell 各组结果与图像（非当前论文结果）

### 8.1 S1：策略对比

| Policy | Harmed-session added WCT mean ± 95% CI | Harmed fraction | All-session mean |
| --- | ---: | ---: | ---: |
| AegisKV | 213.6 ± 72.3 ms | 4.17% | 8.9 ms |
| Host checkpoint | 213.6 ± 72.3 ms | 4.17% | 8.9 ms |
| Neighbor replica | 213.6 ± 72.3 ms | 4.17% | 8.9 ms |
| Recompute | 450.2 ± 110.1 ms | 4.17% | 18.8 ms |

Paired AegisKV - recompute 为 `-236.6 ± 109.9 ms`，在该 modeled deployment 中对应 52.5%
harmed-session mean reduction。三个保护策略相同，是因为配置中的 committed-copy 恢复路径使用相同
effective bandwidth；实验没有人为制造 tier-speed 差异。

![S1 policy WCT CDF](results/20260829T064719Z_paper_complete_v2/figures/s1_policy_wct_cdf.png)

数据：[`s1_policy_wct_cdf.csv`](results/20260829T064719Z_paper_complete_v2/figure_sources/s1_policy_wct_cdf.csv)

### 8.2 S2：负载与故障域

无故障 cells 的 failure-added WCT 均为零。注入故障后的 paired AegisKV - recompute：

| Failure | Load | Paired effect mean ± 95% CI |
| --- | ---: | ---: |
| node_1 | 0.5 | -225.4 ± 103.7 ms |
| node_1 | 1.0 | -236.6 ± 109.9 ms |
| node_1 | 1.5 | -336.2 ± 205.1 ms |
| worker_1 | 0.5 | -225.4 ± 103.7 ms |
| worker_1 | 1.0 | -236.6 ± 109.9 ms |
| worker_1 | 1.5 | -298.5 ± 173.1 ms |

随着 load 增加，recompute 的受损 WCT 增长更快；AegisKV 保持负 paired effect，但高负载区间也更宽。

![S2 load and failure heatmap](results/20260829T064719Z_paper_complete_v2/figures/s2_load_failure_heatmap.png)

数据：[`s2_load_failure_heatmap.csv`](results/20260829T064719Z_paper_complete_v2/figure_sources/s2_load_failure_heatmap.csv)

### 8.3 S3：模型与 projected context

| Model | Context | Harmed-session p95 added WCT mean ± 95% CI |
| --- | ---: | ---: |
| Qwen3-8B | 8,192 | 146.8 ± 27.9 ms |
| Qwen3-8B | 16,384 | 213.6 ± 72.3 ms |
| Qwen3-8B | 32,768 | 445.7 ± 199.1 ms |
| Qwen3-8B | 40,960 | 580.7 ± 274.6 ms |
| Qwen3-32B | 8,192 | 213.6 ± 79.7 ms |
| Qwen3-32B | 16,384 | 273.6 ± 121.7 ms |
| Qwen3-32B | 32,768 | 664.3 ± 380.3 ms |
| Qwen3-32B | 40,960 | 1,045.8 ± 565.4 ms |

结果符合 KV footprint 随 context 和模型规模增长的机制预期，但长 context 的跨 seed 方差明显扩大。

![S3 model and context](results/20260829T064719Z_paper_complete_v2/figures/s3_model_context.png)

数据：[`s3_model_context.csv`](results/20260829T064719Z_paper_complete_v2/figure_sources/s3_model_context.csv)

### 8.4 S4：三算法消融

| Policy | Maximum recovery lateness mean ± 95% CI |
| --- | ---: |
| AegisKV | 239.2 ± 58.7 ms |
| AegisKV DRAM-only | 239.2 ± 58.7 ms |
| AegisKV no AIMD | 239.2 ± 58.7 ms |
| AegisKV no Slack/FIFO | 330.7 ± 53.4 ms |

no-priority penalty 为 `91.5 ± 39.5 ms`。这证明 causal recovery ordering 在受控 fan-in stress 下
有效；四种策略的 total WCT 接近，因此不能把该组写成端到端 latency 大幅提升。

![S4 ablation](results/20260829T064719Z_paper_complete_v2/figures/s4_algorithm_ablation.png)

数据：[`s4_algorithm_ablation.csv`](results/20260829T064719Z_paper_complete_v2/figure_sources/s4_algorithm_ablation.csv)

### 8.5 S5：HBMQuotaAIMD

| Load | Minimum alpha | Maximum alpha | Mean alpha |
| --- | ---: | ---: | ---: |
| 0.75 | 0.500 | 0.759 | 0.613 |
| 1.00 | 0.500 | 0.668 | 0.565 |
| 1.50 | 0.451 | 0.568 | 0.505 |

负载越高，AIMD 越倾向于降低 Peer-HBM quota。审计还确认 no-AIMD 控制组的 alpha range 为零。
该组验证控制律与状态变化，不单独证明端到端性能收益。

![S5 AIMD timeline](results/20260829T064719Z_paper_complete_v2/figures/s5_aimd_timeline.png)

数据：[`s5_aimd_timeline.csv`](results/20260829T064719Z_paper_complete_v2/figure_sources/s5_aimd_timeline.csv)

### 8.6 S6：Prefix reuse

| Reuse bin | Paired sessions | AegisKV - recompute mean ± 95% CI |
| --- | ---: | ---: |
| `<90%` | 83 | 0.0 ± 0.0 ms |
| `90--99%` | 129 | -16.0 ± 9.7 ms |
| `>=99%` | 28 | -14.4 ± 32.5 ms |

`90--99%` 分组显示小幅改善；`>=99%` 分组样本较少且置信区间跨零，不能宣称高 reuse 一定带来
显著收益。`<90%` 为零是当前选中 recovery prefix/action 下的真实 null result。

![S6 prefix reuse](results/20260829T064719Z_paper_complete_v2/figures/s6_prefix_reuse.png)

数据：[`s6_prefix_reuse.csv`](results/20260829T064719Z_paper_complete_v2/figure_sources/s6_prefix_reuse.csv)

### 8.7 S7：带宽与 repair horizon

| Repair | 10 GB/s | 20 GB/s | 25 GB/s |
| ---: | ---: | ---: | ---: |
| 1 s | 8,972.4 ± 6,037.0 | 8,281.7 ± 6,242.9 | 7,825.7 ± 5,693.8 |
| 5 s | 13,439.4 ± 7,285.8 | 12,739.4 ± 6,637.5 | 11,666.5 ± 5,972.8 |
| 15 s | 18,244.9 ± 6,423.5 | 18,371.2 ± 4,892.0 | 17,556.0 ± 4,255.1 |
| 30 s | 18,244.9 ± 6,423.5 | 18,371.2 ± 4,892.0 | 17,556.0 ± 4,255.1 |

数值为 all-session p95 added WCT mean ± 95% CI，单位 ms。带宽增大通常降低恢复影响，repair
horizon 增大则显著恶化结果；但 median across-seed CV 为 60.4%，趋势噪声较大。

15 秒和 30 秒在所有带宽下逐 cell 完全相等。原始事件显示 replacement route 已在 worker repair
之前激活，且没有 degraded-route rebind，因此更长 outage 对有限 trace 不再 binding；这不是绘图聚合错误。

![S7 bandwidth and repair](results/20260829T064719Z_paper_complete_v2/figures/s7_bandwidth_repair.png)

数据：[`s7_bandwidth_repair.csv`](results/20260829T064719Z_paper_complete_v2/figure_sources/s7_bandwidth_repair.csv)

### 8.8 S8：故障数量与故障域

| Failure | Failed workers | AegisKV all-session p95 | Recompute all-session p95 | Paired AegisKV - recompute |
| --- | ---: | ---: | ---: | ---: |
| worker_1 | 1 | 5,858.2 ± 3,624.5 | 5,471.5 ± 3,667.0 | 386.7 ± 802.0 ms |
| worker_2 | 2 | 15,878.5 ± 5,021.9 | 15,401.4 ± 5,312.4 | 477.1 ± 1,164.6 ms |
| worker_4 | 4 | 52,445.4 ± 6,866.3 | 54,750.5 ± 6,498.5 | -2,305.1 ± 6,086.1 ms |
| node_1 | 2 | 18,371.2 ± 4,892.0 | 14,452.7 ± 4,563.6 | 3,918.5 ± 7,545.4 ms |

AegisKV 只有 1/4 场景的端到端均值更低，4/4 paired 95% CI 都跨零，因此这组结果不支持
“AegisKV 在所有故障域全面降低 WCT”。恢复 lateness 的改善更一致，说明当前瓶颈还包括 replacement
route、queueing 和大范围并发恢复，而不仅是单个 KV extent 的恢复代价。

![S8 failure domain and count](results/20260829T064719Z_paper_complete_v2/figures/s8_failure_domain_count.png)

数据：[`s8_failure_domain_count.csv`](results/20260829T064719Z_paper_complete_v2/figure_sources/s8_failure_domain_count.csv)

### 8.9 S9：Transfer model verification

18 个 cells 同时覆盖 Qwen3-8B/32B、10/20/25 GB/s 和 DRAM/Peer-HBM recovery。最大
analytical-versus-DES absolute error 为 `0.000000440 ms`。

这证明 transfer 公式与 DES 实现一致，不证明 10/20/25 GB/s 是真实 A100 end-to-end 实测，也不解除
service-time calibration blocker。

![S9 transfer verification](results/20260829T064719Z_paper_complete_v2/figures/s9_transfer_model_verification.png)

数据：[`s9_transfer_model_verification.csv`](results/20260829T064719Z_paper_complete_v2/figure_sources/s9_transfer_model_verification.csv)

### 8.10 E1：AgentX/Weka 外部有效性

| Policy | Harmed-workflow added WCT mean ± 95% CI |
| --- | ---: |
| AegisKV | 650.4 ± 200.1 ms |
| Recompute | 790.4 ± 130.8 ms |

Paired all-workflow AegisKV - recompute 为 `-70.0 ± 90.9 ms`。10 个 workflow-sampling seeds 中
3 个改善、7 个相同、0 个恶化，但总体置信区间跨零。

E1 中实际 recovery action 包含 DRAM 和 RECOMPUTE；AegisKV 触发 DRAM recovery 的 seeds 为
3/10，recompute baseline 为 7/10。因为所有安全 Slack LCB 都为零，任何差异只能归因于 Algorithm 2
已经 committed 的 protection extent，不能归因于 Algorithm 1 recovery priority。

![E1 AgentX external validity](results/20260829T064719Z_paper_complete_v2/figures/e1_agentx_external_validity.png)

数据：[`e1_agentx_external_validity.csv`](results/20260829T064719Z_paper_complete_v2/figure_sources/e1_agentx_external_validity.csv)

## 9. 理论预期与实际结果对照

| 理论/设计预期 | 当前结果 | 判定 |
| --- | --- | --- |
| 已提交的 KV prefix 应比完整 recompute 更快 | S1 harmed-session mean 降低 52.5% | 支持，限 modeled timing |
| Slack priority 应优先恢复关键 workflow | S4 lateness penalty `91.5 ± 39.5 ms` | 支持，限 controlled stress |
| 高 load/大 context/大模型提高故障代价 | S2/S3 总体符合，CI 随规模增大 | 支持趋势 |
| AIMD 应在压力下收缩 Peer-HBM quota | S5 高负载 alpha 更低 | 支持机制 |
| 更高 reuse 应产生更大收益 | S6 仅 90--99% 显著，>=99% CI 跨零 | 部分支持/样本不足 |
| 更高带宽应降低恢复影响 | S7 通常下降，但 CV 高 | 弱支持 |
| AegisKV 应在所有故障规模下占优 | S8 仅 1/4 均值更优，全部 CI 跨零 | 不支持 blanket claim |
| Algorithm 1 应在 observed DAG 上产生优先级收益 | E1 safe Slack 全零 | 当前不支持 |
| Transfer 公式与模拟事件应一致 | S9 最大误差 4.40e-7 ms | 支持实现一致性 |

因此，当前图像不能统一描述为“与理论完全一致”。更准确、也更能经受审稿的表述是：

1. 保护路径、恢复排序和 AIMD 三类核心机制都被实际执行并通过针对性验证；
2. 在中央 modeled deployment 中，AegisKV 对单 worker failure 的 conditional harm 有明确改善；
3. 大故障域、observed DAG 和高复用尾部仍存在 null/weak result；
4. 当前最重要的纯模拟研究问题已经从“模拟器是否执行机制”转为“Algorithm 1 的保守区间是否
   过宽、多 worker recovery scheduling 是否成为瓶颈，以及 scale/healthy-overhead/capacity 三条
   缺失轴是否会改变当前结论”。

## 10. 配置来源与证据等级

旧 [`research/parameter_provenance.json`](research/parameter_provenance.json)、
[`research/reviewer_configuration_audit.md`](research/reviewer_configuration_audit.md) 和
[`research/source_registry.json`](research/source_registry.json) 主要服务第 6--9 节的历史 aggregate
lineage，不能自动升格为 v3 GlobalDES 参数权威。v3 的实际来源和缺口如下：

| 配置 | 当前来源/冻结方式 | 证据等级与限制 |
| --- | --- | --- |
| TraceLab request/timing | v0.0.2 DuckDB + 精确 SHA-256；compiler 重审 cardinality/schema/异常集 | artifact observation；context/load 是 modeled projection |
| P6 prefix identity | 可重生 `controlled_prefix_reuse-v1.json` + 精确 SHA-256 | deterministic synthetic；`empirical_measurement=false` |
| P11 Weka/AgentX | 1.847 GB source、compiled manifest/requests/predictor 四类 hash | observed workload + modeled causal predictor；0.928571 simultaneous coverage 未达 0.95，尚未运行 P11 |
| A100 集群/资源图 | `configs_v3/common/deployment.json` 中的 4×8 A100-40GB 与逐资源日历 | controlled modeled deployment，不是本机物理 inventory |
| Qwen3 geometry/KV bytes | deployment JSON 中的 model geometry、派生式以及 8B/32B frozen revisions | model revision gate 已解除；正式 run 仍需实际 model path、simulator/timing lineage |
| HBM/DRAM budgets | 6 GiB/GPU Aegis HBM；512 GiB/node Aegis DRAM；P5/P10 显式 sweep | preregistered modeled capacity，不是 observed occupancy |
| Copy bandwidth/setup | NVLink/PCIe/NIC/fabric/DRAM 显式字段及 P7/P12 axis | modeled and not same-topology calibrated |
| Scheduler/forward timing | run-owned timing class/source/SHA-256 与 simulator implementation hash | 只能标 `calibrated` 或显式 `modeled_not_calibrated`；缺失应 fail-close |
| Failure/repair | 10 s deterministic trigger、30 s horizon、100/50 ms 受控参数 | modeled；v3 无 MTTF/field-hazard claim |

每个新 lineage 必须保存 resolved 三层 JSON、artifact/model/simulator/timing hashes、runtime bindings 和
raw event tables。只有配置文字、本机 cache 路径或一个可运行 model ID 都不足以建立不可变 lineage。

## 11. SGLSim 证据边界与未覆盖实验

审计强制保留以下限制：

1. 当前 scheduler/forward 和 copy timing 不是 4×8 A100-40GB 同拓扑实测；旧 aggregate 线性系数不是 v3 运行权威；
2. 100 ms failover 和 50 ms routing outage 没有在该拓扑校准；
3. TraceLab 使用 context projection 和 load-controlled arrivals；
4. P6 是可重生 controlled-prefix artifact，不是 TraceLab observed content reuse；
5. completion/eviction 到物理 Aegis extent 的 release chain 已在代码中修复，154-test 联合回归通过；
   但 104732 的 7-completed/1-scheduled 请求终态聚合 bug 使 post-fix lineage 无效。必须先修聚合器，再由
   新 immutable lineage 同时通过 8/8 terminal、terminal-zero 和 conservation audit；
6. worker/GPU/process failure 当前仍以 DP worker 为主要清除单位；TP>1 下单张物理 GPU/process 的 shard
   fail-stop 尚未端到端实现；
7. P0 v4 已覆盖 controller + bridge 合同 cell，但 target-before-source failure 后的
   route-bound standby replacement/rotation 尚未完整进入 full-Scheduler offline 执行链；
8. replay 尚未严格绑定 Router 已提交 output position；缺少 committed-output-position-aware silent replay；
9. 普通 HiCache/radix H2D/D2H/prefetch/eviction 还没有全部以显式 `TransferRecord` lifecycle 进入 GlobalDES；
10. recovery 虽有 job/plan/outcome artifact，但尚未统一为与 protection 同构的 recovery `TransferRecord`；
11. P11 compiled artifact 已建立，但 95% workflow-simultaneous coverage 为 0.928571，低于 0.95，且
    safe-Slack LCB 非零数为 0；历史 E1 不是 P11；
12. P5 冻结为三个独立 zero-origin phase cells，不能表示同一 controller 跨 phase 携带状态；
13. 目标 GlobalDES 显式模拟每 GPU/node ledger 和 physical transfer jobs，但仍不分配真实 HBM/DRAM，
    不移动 KV tensor；
14. P1/P12 代表性 smoke 使用 1 request/cell 和 `mock-model`；P1 故障未命中 active request，P12
    只验证 protection formula，且 `strict_extent_accounting=false`；
15. P4 101529 smoke/plots 是 pre-fix 诊断；failure 未命中 active request，不能用于 P4 消融或资源结论；
16. code tests、runtime compilation、P0 contract 和 smoke runner 分别是不同证据层，任何一层都不能单独建立
    P1--P12 performance claim。

因此这些图是 **simulation/mechanism evidence**。它们可以用于撰写模拟评价、机制分析和研究假设，
但图注与正文必须明确写 `SGLSim offline global-virtual-time simulation`，不能把
modeled latency 表述成设备实测值，也不能把 smoke 图当作论文 performance figure。

## 12. 复现当前 SGLSim 实验

以下命令分别复现合同、代码测试和代表性 smoke；它们不是“完整论文实验”的同义词。

### 12.1 复现 P0 contract-only gate 与诊断图

```bash
P0_RUN_DIR=/tmp/aegiskv-global-des-p0-reproduction
uv run --extra test python AegisKV/experiments/run_global_des_p0.py \
  --run-dir "$P0_RUN_DIR"
uv run --extra test python AegisKV/experiments/plot_global_des_p0.py \
  --p0-dir "$P0_RUN_DIR/P0"
```

目标目录必须事先不存在；runner 拒绝覆盖旧 lineage。输出仍只是 contract-only/
non-performance。

### 12.2 复现当前代码级证据

GlobalDES/Scheduler/resource/physical-transfer/hard-failure 回归：

```bash
TMPDIR=/tmp uv run --extra test pytest \
  test/unit/srt/test_offline_aegis_transfer_bridge.py \
  test/unit/srt/test_offline_cluster_aegis_physical_transfers.py \
  test/unit/srt/test_offline_cluster_hard_failstop.py \
  test/unit/srt/test_offline_cluster.py \
  test/unit/srt/test_offline_resource_bridge.py \
  test/unit/srt/test_offline_coordinator.py \
  test/unit/srt/test_global_des.py \
  test/unit/srt/test_data_parallel_controller.py \
  test/unit/srt/test_time_driver.py \
  test/unit/srt/test_pipeline_parallel_contract.py -q -x
```

Request/config/runtime compiler/P11 artifact 冻结回归：

```bash
TMPDIR=/tmp uv run --extra test pytest \
  AegisKV/experiments/tests/test_global_des_config.py \
  AegisKV/experiments/tests/test_global_des_workload.py \
  AegisKV/experiments/tests/test_global_des_causal_workload.py \
  AegisKV/experiments/tests/test_global_des_runtime_compiler.py \
  AegisKV/experiments/tests/test_build_weka_observed_workload.py -q
```

P11 artifact 自审计：

```bash
uv run python -m AegisKV.experiments.src.build_weka_observed_workload --audit-only
```

Runner 回归：

```bash
TMPDIR=/tmp uv run --extra test pytest \
  test/unit/srt/test_global_des_suite_runner.py -q
```

测试通过数会随新增 gate 变化，应引用本次命令输出而非本报告中的旧固定数字。无论数量多少，测试和
2,134/2,134 runtime compilation 都不是 performance cells。

### 12.3 复现代表性 smoke cells

先只做选中 cell 的 readiness 检查：

```bash
TMPDIR=/tmp uv run --extra test python -m \
  AegisKV.experiments.src.global_des_suite_runner \
  --cells P1:0 P4:0 P11:0 P12:0 --check \
  --model-path qwen3-8b=mock-model \
  --model-revision qwen3-8b=b968826d9c46dd6066d109eabc6255188de91218
```

只选择已通过 readiness 的 P1/P12 ordinal 0，并限制 1 request/cell：

```bash
SMOKE_RUN_DIR=/tmp/aegiskv-global-des-p1-p12-smoke
TMPDIR=/tmp uv run --extra test python -m \
  AegisKV.experiments.src.global_des_suite_runner \
  --smoke --groups P1 P12 --max-requests-per-cell 1 \
  --run-dir "$SMOKE_RUN_DIR" \
  --model-path qwen3-8b=mock-model \
  --model-revision qwen3-8b=b968826d9c46dd6066d109eabc6255188de91218
```

`mock-model` 只授权 modeled smoke，目录也必须事先不存在。已固化的参考输出是第 0.2 节的
`20260829T100844Z_global_des_scheduler_smoke`。

101529 P4 artifact 的历史命令与 manifest 保留供审计，但当前代码修复后重新运行必须写入新目录；
不得覆盖旧 lineage，也不得把旧四张 plots_v3 图当作修复验证。

### 12.4 为什么本报告不给出完整 `--all` 执行指令

配置层已经不再是主要阻断：Qwen3-32B revision、P4 causal workload、P11 observed manifest、
no-Algorithm-1 adapter、P3 trigger scaling 和 per-node DRAM quota 均已绑定；带显式 frozen modeled
simulator/failure bindings 时 P1--P12 2,134/2,134 可编译。

但现在不应启动或宣传 `--all` 为正式论文运行，因为第 11 节的 request terminal aggregation、单物理
GPU/process fail-stop、full-Scheduler target-before-source route rotation、committed output position replay、普通 HiCache
transfer lifecycle 和 recovery `TransferRecord` 仍未闭合；P11 predictor gate 也未达 0.95。即使 runner 可以开始
执行，所得 full matrix 也不能通过严格正确性/证据审计。在这些门槛解除前给出“一键全套论文结果”命令
会把 runtime compilability 或 modeled smoke 误导为 paper execution。

## 13. 结论与下一轮纯模拟工作

当前状态不再是“只有 aggregate prototype”：GlobalDES 单堆、真实 Offline Scheduler、逐 GPU HBM/
逐 node DRAM ledger、竞争资源日历、独立 HBM/DRAM protection commit、hard fail-stop 和
physical recovery job 已在代码中接通。P0 v4 contract gate 为 12/12 PASS，其中
target-before-source controller + bridge rotation cell 的 4/4 assertions 与 10/10 invariants PASS；
它必须继续保留 `full_scheduler_integration=false`、
`p1_p12_execution_ready=false`、`performance_result=false`，不能用后续代码追改。

新 runner 的 P1/P12 各1-request smoke 只建立部分 integration evidence：P1 failure 未命中 active
request，P12 是 2 个 modeled protection jobs 的 exact formula check。P4 8-request smoke 虽然记录了
46 个 protection jobs 和逐 GPU/node timeline，却是 completion/eviction extent release 修复前的
lifetime-leak artifact。代码修复已通过 154-test 联合回归，但 104732 因 7/8 terminal aggregation bug
仍无效。P11 102845 完整执行了一个 57-request recompute workflow，却没有 protection/recovery 或 paired
policy。它们都正确标为 `paper_ready=false`，不是论文 performance result。
正式 P1--P12 仍为 **0 cells**，完整 2,146-cell suite 未运行。

下一轮必须依次完成：

1. 修复 request terminal-state aggregation，并用新 P4 lineage 同时验证 8/8 terminal、
   terminal-zero/conservation 和 manifest/audit 一致；
2. 补全单物理 GPU/process fail-stop、full-Scheduler target-before-source route rotation、committed output position silent
   replay、普通 HiCache transfer lifecycle 和 recovery `TransferRecord`；
3. 对 P11 改进或重新校准 admission-only interval，达到预注册 0.95 workflow-simultaneous coverage；若仍
   safe Slack 全零，必须把 Algorithm 1 结论保守地报告为 null；
4. 运行能命中 active request 的 hard-failure/recovery smoke，并将 strict extent、capacity、conservation、
   causal/replay audit 写入新 immutable integration gate lineage；
5. 门槛通过后再运行完整 P1--P12，从同一 lineage 绘制 WCT/TTFT/TPOT、per-GPU HBM、
   per-node DRAM、transfer Gantt 和 failure zoom-in。

历史 S1--S9+E1/613-cell aggregate 结果继续留档，但不承担新论文结果角色。在上述门槛
完成前，可以说“目标 simulator 的核心代码链已实现并有集成测试”，不能说“完整论文模拟和
结果已完成”。新模拟器设计、开源调研、事件/输出合同和完成门见
[`GLOBAL_VIRTUAL_CLOCK_DESIGN.md`](GLOBAL_VIRTUAL_CLOCK_DESIGN.md)，开源候选的逐项审计见
[`OPEN_SOURCE_SIMULATOR_RESEARCH.md`](OPEN_SOURCE_SIMULATOR_RESEARCH.md)。
