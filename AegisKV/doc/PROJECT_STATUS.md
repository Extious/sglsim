# AegisKV project status and thread consolidation

更新时间：2026-08-29  
工作区：`/home/comp/zhanzhao/sglsim`，分支 `aegiskv`

本文整理以下两个 Codex thread，并用当前 checkout 中的代码、manifest 和 README 复核其最终状态：

- `01a01dae-f8df-7131-a3d2-bddfc9fa83a3`：SGLSim AegisKV 实现
- `01a005a8-95b8-7c41-af1b-50d06b711944`：Mooncake AegisKV 实现

本文是导航与状态台账，不替代 [current_design.md](design/current_design.md)、实验审计或原始 thread。

## 1. 总结

两个 thread 最终收敛到同一个设计：`source_owned_independent_hbm_dram_v2`。HBM 与 DRAM 分别准入、分别从 Source Worker 复制、分别提交和失败；允许 HBM-only，并彻底移除 target-owned HBM→DRAM writeback/destage。Algorithm 1 负责因果 Slack，SlackLandlord 管两个独立 pool，HBMQuotaAIMD 只调整 Peer-HBM 配额。

Mooncake thread 完成了跨 Application、Router、vLLM 和 Mooncake Controller 的实现原型与多轮静态/物理 pilot，但尚未形成可独立复现的 current-design 正式硬件证据。SGLSim 已新增 `sglsim_complete_v1` 统一 metadata DES，把 target-first rotation、真实 replacement HBM pool、source fail-stop、queue outage/rescheduling、fence/rebind 和 silent replay 与三算法接入同一连续时间事件流；complete-path demo audit 为 PASS。历史大规模 campaign 与 preliminary WCT campaign 的原证据边界保持不变，硬件校准后的正式 replay 仍被 33 个证据缺口阻塞。

因此目前最准确的状态是：**完整 current-design 控制/故障语义 metadata simulation 已有统一可执行入口；三层 JSON 驱动的 S1--S9 + E1 统计复跑已完成并生成 10 组图；真实 Mooncake current-design end-to-end、同拓扑 A100 校准与 paper-ready 结论仍未完成。**

2026-08-29 的当前正式模拟入口是
[`../experiments`](../experiments/README.md)，lineage 为
`20260829T064719Z_paper_complete_v2`：613/613 cells、341,772 requests、18,376
workflow/session observations、10 PNG + 10 PDF，语义与复现审计 0 failed checks。
E1 读取 pinned 393-workflow AgentX/Weka 源文件并保留真实 DAG、相对 request 时序和
prefix hash；20 个 paper workflow 在 seed 间无放回且零重叠。该 campaign 仍是 offline
metadata DES，状态为 `PASS_WITH_PHYSICAL_CALIBRATION_BLOCKER`。
该 lineage 还把默认 prefix/decode chunk 与权威 `current_design.md` 对齐为
4,096/256 tokens，并在审计中逐组验证；S9 的 1,024-token prefix 仅是标注清楚的
transfer-formula 微验证覆盖。上一版 `20260829T055857Z_paper_complete_v2` 因默认使用
8,192-token prefix chunk 被本 lineage 取代。中间版
`20260829T062726Z_paper_complete_v2` 的结果 CSV 与图像数据虽然一致，但其预执行研究
快照残留了更早 E1 run 的动态结果数字，也已被当前 lineage 取代。

## 2. SGLSim thread 内容

### 2.1 第一阶段：同步设计并建立机制原型

thread 首先读取 Mooncake 任务，发现本项目旧文档仍要求“Peer-HBM 以 DRAM 为前置”。随后新增：

- [aegis_independent_tiers.py](../python/sglsim/srt/simulation/aegis_independent_tiers.py)：独立 HBM/DRAM 状态、pool-local SlackLandlord、path-local commit、residency、target rotation 与 HBMQuotaAIMD。
- [aegis_independent_tier_manager.py](../python/sglsim/srt/managers/aegis_independent_tier_manager.py)：接入 SGLSim Engine、DP Controller、Scheduler/PP GroupCommit 的适配层。

早期合成机制实验曾报告相对 recompute 的 18.6% Slack-visible harm 降低、相对 static independent 的 5.6% 降低，见 [experiments/latest_design](experiments/latest_design/README.md)。后续自审计发现该阶段尚未完整接入渐进 seal、独立传输事件、最终回收、完整故障路径和 Algorithm 1，且策略间复制工作量不完全匹配。因此这些数字只保留为历史机制原型，不是当前完整实现或论文结论。

### 2.2 第二阶段：真实 Weka metadata replay 与 baseline

thread 随后补齐因果 GBDT→CPM Slack、渐进 seal、独立 path event、fence/rebind、silent replay、PP/TP barrier 与终态回收，并将完整 393-workflow Weka corpus 映射到 SGLSim：98,827 requests、21.64B input tokens、106.47M output tokens。

这一阶段同时审计本地 related-work 语料并实现 capacity-matched baseline adaptations，包括 DejaVu、KevlarFlow、GhostServe、RaidServe/FailSafe、LUMEN/dispatch 类控制等。公平对照显示 AegisKV 并不总是最优：保护覆盖不足、Peer-HBM recovery 与 recompute 成本过近、Algorithm 1 误差较大，而 failure-time load-aware dispatch 是更主要的瓶颈。该负结果促使后续把恢复调度、长上下文校准和工作量匹配列为重点。

后来确认当时所谓 Peer-HBM 物理系数实际来自 Mooncake TCP，而不是 CUDA P2P/D2D；4090→DP8 外推和超长上下文也越过校准范围。因此 [real_weka_full_design/RESULTS.md](experiments/real_weka_full_design/RESULTS.md) 与 [real_weka_baselines/README.md](experiments/real_weka_baselines/README.md) 已将该组结果标为 `INVALIDATED FOR PERFORMANCE CLAIMS`。

### 2.3 第三阶段：fail-closed 物理 profile consumer

SGLSim thread 随后建立严格的 V100/A100 producer-consumer 边界：在创建结果目录、读取 trace 或拟合模型前，验证硬件标签、source/config/profile SHA、四个独立 protection/recovery rate、wire enum、四种 residency、独立 ACK、无 target-owned/HBM→DRAM、HBM_ONLY 直接回收及 AIMD trace。

主要入口位于：

- [real_weka_full_design/run.py](experiments/real_weka_full_design/run.py)
- [import_mooncake_v100_profile.py](experiments/real_weka_full_design/import_mooncake_v100_profile.py)
- [a100_dp2_contract.py](experiments/real_weka_full_design/a100_dp2_contract.py)
- [aegis_mooncake_bridge.py](../python/sglsim/srt/simulation/aegis_mooncake_bridge.py)
- [aegis_output_contract.py](../python/sglsim/srt/simulation/aegis_output_contract.py)

截至当前 checkout，[eurosys-paper-readiness.json](experiments/real_weka_full_design/eurosys-paper-readiness.json) 仍为 `BLOCKED_PAPER_REQUIREMENTS`：33 gaps、0 planned cells，所有 replay/result/performance/formal/paper authorization 均为 false。历史 queue、monitor、hash 和测试记录是静态控制证据，不是 runtime 或性能证据。

### 2.4 第四阶段：当前三套 SGLSim campaign

当前项目中有三套用途不同的 simulation lineage：

1. [sglsim_complete_v1](experiments/sglsim_complete_v1/README.md) 是现行完整控制/故障语义的统一 metadata DES。它在同一事件流中运行因果 Slack、渐进 seal、独立 HBM/DRAM、per-target AIMD、target-first rotation、source fail-stop、recovery priority、queue outage/rescheduling、route rebind 和 silent replay。complete-path demo 的严格 audit 为 PASS；这证明统一路径可执行，不等于大规模统计或物理结果。
2. [sglsim_large_scale_v2](experiments/sglsim_large_scale_v2/README.md) 是冻结的大规模 metadata DES。它覆盖 33 scenarios、311 policy-scenarios、3,110 paired policy-seed runs，并保留 baseline、三算法消融、并发/故障率、模型/规模、带宽、容量、alpha、reuse 和 recovery 诊断。当前 full audit 为 PASS，数字可在其 simulation-only contract 内使用；但它不执行 workflow failure WCT，也没有完整 standby rotation，不能把其旧数字改名为 complete-v1 结果。
3. [sglsim_wct_preliminary_v1](experiments/sglsim_wct_preliminary_v1/RESULTS.md) 执行本地 DES workflow completion time，形成 S1–S9 preliminary 图。它包含 10-seed architecture/concurrency campaigns、Qwen3-8B/32B、消融、alpha/occupancy/reuse/topology 和独立 RTX4090 timing holdout；仍标记 `paper_ready=false`。部分结果为 modeled timing、共享样本、弱/负效应或未通过 all-cell calibration gate，不能选择性删去。

第四套、也是当前默认入口，是根目录下的
[`../experiments`](../experiments/README.md)。它把 request/failure/deployment 三层配置分别
固化为 JSON，统一调用 complete metadata DES，重新执行 S1--S9，并增加 E1 公共
AgentX/Weka observed-DAG 外部有效性。每个 lineage 同时快照配置、研究依据和执行源码；
旧 `doc/experiments` 内容不被当前 runner 读取。

当前入口还新增了 [`../experiments/physical`](../experiments/physical/README.md) 作为物理
校准 handoff。它把最终模拟 lineage 的全部非 seed 实验轴压缩为 3-seed、201-cell matched
计划，绑定配置、代码、raw/shadow artifact 和 current-design 哈希；历史 2xA100-80GB
inventory 严格 preflight 为 `BLOCKED_TOPOLOGY_OR_IDENTITY`。该结果只证明 gate 正常拒绝
不匹配资源，没有运行或授权物理 replay。

## 3. Mooncake thread 内容

### 3.1 分支与实现

Mooncake thread 在 `/home/comp/zhanzhao/Mooncake` 创建 `codex/aegiskv` 分支，并实现了：

- `mooncake-store/include/aegiskv_controller.h`
- `mooncake-store/src/aegiskv_controller.cpp`
- `aegiskv/src/aegiskv/application/algorithm1.py`
- `aegiskv/src/aegiskv/router/server.py`
- `aegiskv/src/aegiskv/vllm/gate.py`
- `aegiskv/src/aegiskv/vllm/hbm.py`

实现保持 Mooncake 所有权边界：framework cache manager 拥有 request/GPU-HBM mapping，Transfer Engine 移动 bytes，Store Master 管全局 object/replica metadata，Store Node 管本地 layout/eviction。legacy target-owned writeback API 显式失败，Source Worker 独立提交 DRAM。

早期记录的验证为 Python 290 passed/1 skipped、native Controller/RPC/Admin 3/3、compileall 与 diff check PASS。这些数字描述当时 checkout，不等于本轮重新执行的 current verification。

### 3.2 物理 pilot 与设计修订

单节点 2×RTX4090、Qwen3-8B、Router-DP=2 的三 seed pilot 在 failed-wave 上出现小幅正收益，但 C=8 转负，且 whole-workload mean WCT 在所有并发点都变差。独立两节点 PP=2 pilot 报告约 6.4% failure-added WCT 降低；两组 evidence 不可合并，均为 preliminary。

后续审计发现部分 Peer-HBM 路径实际使用 Mooncake TCP，不是 CUDA P2P/D2D，旧 4090 校准因此不能进入正式 SGLSim replay。thread 转而建立 A100/V100 exact-source、model/config、registered-copy、recompute、transfer、schema-v2 profile、four-condition mechanism gate 和 paired-effect 的 fail-closed chain。大量作业只到 static、queued-zero-runtime 或 resource-blocked 状态，不能被拼接为完整物理 lineage。

### 3.3 论文与图表收口

2026-08-26/27，用户把重点转为 SGLSim overnight evaluation、Mooncake 4090 pilot 和 R1–R6/S1–S9 图表。最终 Mooncake 侧生成：

- `/home/comp/zhanzhao/Mooncake/output/pdf/aegiskv-preset-vs-real-side-by-side.pdf`
- `/home/comp/zhanzhao/Mooncake/output/pdf/aegiskv-experiment-plan-preset-vs-observed.pdf`
- `/home/comp/zhanzhao/Mooncake/aegiskv/experiment_plan/figure-gap-audit.md`

最新 thread 结论是 15/15 图通过视觉门槛，但 0/15 paper-ready。R2 仍是语义 gate 失败的诊断；R3/R4 样本量有限；R5/R6 主要是 mechanism gate；SGLSim panels 是 modeled/preliminary；S9 为 15/16 calibration cell 通过。当前 Mooncake EuroSys 正文仍包含旧 evaluation，尚未同步最新图包，因此不可直接投稿。

SGLSim 已只读复制 S1–S9 到 [paper/evaluation-assets](paper/evaluation-assets/README.md)，并生成 [插入验证 PDF](../output/pdf/aegiskv-sglsim-figure-insertion-proof.pdf)。这些是稳定 include targets，不提升原始 evidence level。

## 4. 现行设计契约

| 主题 | 当前规则 | 已废弃规则 |
|---|---|---|
| HBM/DRAM 准入 | 两层独立运行 SlackLandlord | Peer-HBM 必须先有 DRAM |
| 复制所有者 | Source Worker 分别 D2D/P2P 与 D2H | target 接管并 HBM→DRAM destage |
| residency | HBM_ONLY / DRAM_ONLY / HBM_AND_DRAM / BYPASS | 只有 DRAM 或 DRAM+HBM |
| commit | 每层独立 PENDING/COMMITTED/ABORTED | 单个 bundle completion 同时提交两层 |
| AIMD | 只调整 Peer-HBM `alpha` quota | 同时控制 DRAM 或传输速率 |
| HBM 回收 | HBM_ONLY 可直接删除，缺副本时重算 | 必须先写回 DRAM |
| Slack | 因果可见 residual DAG；缺信息置零 | 使用未来完成时间或非因果 oracle |
| 恢复 | 只使用完整连续 committed prefix | 使用 pending 或把两层 token 相加 |

## 5. 当前项目内容与证据台账

| 内容 | 权威入口 | 当前状态 | 能证明什么 | 不能证明什么 |
|---|---|---|---|---|
| 设计 | [current_design.md](design/current_design.md) | current | 目标语义与算法契约 | 已实现、已验证或有性能收益 |
| SGLSim 实现 | `python/sglsim/srt/simulation/aegis_*`、manager adapters | implemented, dirty checkout | metadata 状态机、控制器和集成路径存在 | 与 pinned SGLang 全面等价、物理 KV 移动 |
| 完整语义模拟 | [sglsim_complete_v1](experiments/sglsim_complete_v1/README.md) | complete-path audit PASS | current-design 控制/故障语义在同一 metadata DES 中闭环 | 大规模统计优势、物理性能或 paper-ready |
| 大规模 campaign | [sglsim_large_scale_v2](experiments/sglsim_large_scale_v2/README.md) | full audit PASS | simulation-only estimands、3110 paired runs、完整负结果保留 | 执行式 WCT、standby rotation、物理性能 |
| WCT campaign | [sglsim_wct_preliminary_v1](experiments/sglsim_wct_preliminary_v1/RESULTS.md) | preliminary | 本地 DES WCT 与 S1–S9 机制趋势 | paper-ready、物理 A100 或通用优势 |
| 三层 JSON complete campaign | [../experiments](../experiments/README.md) | 613/613, audit PASS | current-design offline metadata DES、S1--S9 + E1、完整原始 lineage 与 10 组图 | 物理 A100、online serving 实测或 paper-ready |
| 硬件校准 replay | [real_weka_full_design](experiments/real_weka_full_design/README.md) | BLOCKED | fail-closed consumer/readiness | 任何新 replay 或性能结论 |
| 历史 Weka/4090 | [invalidated RESULTS](experiments/real_weka_full_design/RESULTS.md) | invalidated | 诊断与设计演化 | 当前性能 claim |
| S1–S9 论文图包 | [evaluation-assets](paper/evaluation-assets/README.md) | visual insertion-ready | 可复现图副本、caption 与 evidence qualifier | paper-ready empirical evidence |
| SGLSim 论文正文 | [paper](paper/README.md) | draft, not synchronized with new assets | 论文结构和设计章节 | 当前可投稿版本 |
| Mooncake 物理实现 | `/home/comp/zhanzhao/Mooncake/aegiskv` | external implementation/pilot | 工程路径和有界 preliminary evidence | 独立 current-design formal replication |

## 6. 使用与维护规则

1. 先改 `design/current_design.md`，再同步实现、论文算法、System Overview、测试和审计。
2. 新结果必须写入新的 immutable lineage；不得覆盖或重新标记 invalidated 目录。
3. 报告时必须写出 evidence class：design、static test、metadata DES、WCT DES、physical calibration、physical end-to-end 或 paper-ready。
4. S1–S9 图可以用于排版和 preliminary 讨论，但不能省略各自的 modeled/derived/calibration qualifier。
5. `Archived/`、历史 `four_path_v2` 和早期 `latest_design` 仅用于追溯，不应作为“当前最佳结果”入口。
6. 当前工作树有大量用户改动和未跟踪结果；整理时不做批量移动、删除或提交，除非另有明确授权。

## 7. 已发现的当前不一致

- [aegiskv-paper-system-design-v2.drawio](figures/aegiskv-paper-system-design-v2.drawio) 仍包含 `destage`、`Destage executor` 和 `DRAM direct protection or backing`，其 PNG/PDF 导出及 [system-overview-review.md](figures/system-overview-review.md) 也继承了旧 target-owned 语义。它们当前不能被称为与 `current_design.md` 同步。
- `paper/scripts/build.sh` 的 design fingerprint 只约束 `current_design.md` 与 System Design TeX，不审计 Draw.io XML；因此 LaTeX build PASS 不能发现上述图语义漂移。
- [paper/evaluation-assets](paper/evaluation-assets/README.md) 是 2026-08-27 新迁移的 S1–S9 稳定图包，但 `paper/main.tex` 仍未引用它；“图包可插入”和“正文已更新”是两个不同状态。

修复顺序应为：先按 current design 重画并审计 System Overview，再重新导出 PNG/PDF、更新 figure review，最后重建论文；不能只改 README 中的“已同步”措辞。

## 8. 下一步

优先级从高到低：

1. 选择论文的主证据路线：simulation-only paper，或等待 current-design physical profile 后再做 calibrated paper。两条路线不能混用证据标签。
2. 若走 simulation-only 路线，使用已完成的三层 JSON complete lineage 做主证据，并把新图接入 `paper/main.tex`；必须明确不建模物理数据面，且 E1 的 CI 跨零与零 Slack LCB 不得省略。
3. 若走 physical 路线，先完成同一 source/config/hardware lineage 的 model、registered-copy、recompute、六次 transfer、schema-v2、AIMD、four-condition gate 和 paired effect，再解除 `real_weka_full_design` 的 block。
4. 对 complete-v1 与在线 SGLSim Engine 做 pinned-SGLang-grounded runtime parity 审计；统一 DES 的 PASS 仍不等价于在线 Engine 全面 parity。
5. 将 `paper_ready=false` 保持到样本量、校准、端到端语义与正式统计门槛全部满足。
