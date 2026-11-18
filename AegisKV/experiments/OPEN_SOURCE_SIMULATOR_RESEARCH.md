# 面向 AegisKV 全局虚拟时钟实验的开源模拟器调研

> 调研结论（2026-08-29）：目前没有一个开源项目同时提供 **SGLang 调度语义、单一全局 DES、逐 `(DP, PP, TP)` 物理 GPU HBM、逐节点 DRAM、受争用 KV 迁移、GPU/进程/节点故障与恢复**。本项目应继续以 SGLSim 的 SGLang 对齐运行时为语义主体，选择性借鉴其他项目的事件核、资源图和传输站建模，不能直接替换成某一个外部模拟器。

## 1. 选择标准

候选项目按下列能力审计：

1. 是否以一个权威全局事件堆推进虚拟时间；
2. 是否保持 continuous batching、prefill/decode、radix/HiCache 等 serving 语义；
3. 是否把 TP/PP/DP 展开为物理设备，而不只保存一个并行度整数；
4. 是否逐 GPU 记录 weights、runtime、primary/prefix KV、Aegis PENDING/COMMITTED HBM；
5. 是否逐节点记录 HiCache primary 与 Aegis DRAM；
6. 是否建模 copy engine、PCIe、NVLink/NVSwitch、NIC、跨节点 fabric 的争用；
7. 是否支持 GPU、进程、worker、node fail-stop 与 repair；
8. 是否能表达 AegisKV 的 source-owned、HBM/DRAM 独立 admission 和分片组提交。

## 2. 候选项目

| 项目 | 可借鉴部分 | 不能直接满足 AegisKV 的部分 | 本项目定位 |
|---|---|---|---|
| [SGLang Simulator PR #33824](https://github.com/sgl-project/sglang/pull/33824) | 最接近 SGLang scheduler/batch 生命周期；标量逻辑时钟 | 不提供完整逐 GPU/DRAM/迁移争用/故障资源模型 | **调度语义参考**；仍以本机 pinned SGLang 为最终权威 |
| [Vidur](https://github.com/microsoft/vidur) / [论文](https://arxiv.org/abs/2405.05465) | 事件驱动框架、workload 与性能预测器分层 | 不是 SGLang runtime；无 AegisKV 的物理资源与故障提交协议 | **事件核与预测器分层参考** |
| [Inference Lab](https://github.com/doublewordai/inference-lab) / [KV resource graph](https://github.com/doublewordai/inference-lab/blob/main/src/kv_cache/graph.rs) | KV resource graph、路径和带宽争用表达最接近目标 | Rust/vLLM 语义；普通 TP 仍可能聚合；缺少完整硬件故障恢复 | **资源图与路径争用首选参考** |
| [BLIS / inference-sim](https://github.com/inference-sim/inference-sim) / [architecture](https://github.com/inference-sim/inference-sim/blob/main/docs/concepts/architecture.md) | 多级 KV、异步 transfer station、资源约束与 invariant | 非 SGLang；TP 与故障域不足以直接表达本设计 | **传输站和 conservation audit 参考** |
| [LLMServingSim](https://github.com/casys-kaist/LLMServingSim) / [architecture](https://llmservingsim.ai/docs/simulator/architecture) | 全局 cycle、异构计算/内存/网络、TP/PP/DP 映射 | vLLM/硬件协同模拟方向；没有 AegisKV failure protocol | **异构设备与网络映射参考** |
| [ASTRA-Sim](https://github.com/astra-sim/astra-sim) | 分布式训练/推理通信与网络拓扑模拟 | 不包含 serving scheduler、KV ownership 和恢复语义 | 可作为后续网络后端，而非顶层 simulator |
| [SimAI](https://github.com/aliyun/SimAI) | collective 与网络性能预测 | 不负责 KV 生命周期或请求调度 | 可作为 TP collective/网络 latency donor |
| [IBM Opal](https://github.com/IBM/opal-sim) / [KV manager design](https://github.com/IBM/opal-sim/blob/main/wiki/KVCache-manager.md) | 离线 serving simulator 与 KV manager 设计材料 | KV copy/eviction 能力不完整，不能承担论文主实验 | 仅作接口与设计参考 |
| [Frontier](https://github.com/NetX-lab/Frontier) | prefill/decode 分离和 transfer events | replica-level KV；无逐节点 DRAM 和 Aegis failure semantics | 仅借鉴 P/D transfer event 表达 |

## 3. 采用方案

本项目采用混合架构，但不引入第二套调度器：

```text
pinned SGLang control flow
          │ mirror
          ▼
real SGLSim Scheduler metadata effects
          │ rank-local snapshots / Aegis intents
          ▼
single GlobalDES heap (ns)
          │
          ├── per-(dp,pp,tp) GPU HBM ledgers
          ├── per-node DRAM ledgers
          ├── copy-engine / PCIe / NVLink / NIC / fabric calendars
          ├── KV extent + tier-local shard-group commit
          └── GPU/process/worker/node failure and repair
```

具体取舍：

- 保留 SGLSim 的真实 `Scheduler`、`PrefillAdder`、continuous batching、radix/HiCache 元数据副作用；
- 离线实验不启动 HTTP 闭环、不调用 `sleep`，所有 arrival、batch completion、transfer、failure、repair 都进入同一个全局 heap；
- 一个 Scheduler 的 TP 对称视图只用于生成每 TP-rank 的一份 shard 账本，再由父进程展开到物理 GPU，禁止把全 TP 字节重复计入单卡；当前 real-Scheduler 正式模型为 PP1（TP1/TP4），PP2×TP4 只有 metadata contract；
- 迁移资源采用显式 availability calendar；并行任务的 makespan 是完成时间最大值，而不是各 DP 时长求和；
- HBM 与 DRAM 复制由同一 source production buffer 独立发起、独立 admission、独立提交；禁止 Peer-HBM 到 DRAM 的 target-owned destage；
- 普通 HiCache D2H/H2D intent 已经展开为每 PP/TP shard 的物理 transfer，与 AegisKV protection/recovery 共用 copy engine、PCIe、NIC/fabric 和 DRAM lane，并实施全 shard commit、故障 rollback、compute dependency 与 stale-incarnation fencing；
- AegisKV recovery 已经用 canonical `RecoveryLoadGroup`、每 shard `TransferRecord`、目标 HBM hold、分片 adoption、明示 fallback 和 Router-output-to-physical-admission gate 表达，不是只修改一个 recovery flag；
- Router 以 committed output position 为权威生成 route `v+1`、重放 prompt + committed outputs 并 fencing stale route；其 continuation tokens 是合成元数据，不是真实 logits/sampling RNG 续跑；
- failure resolver 只执行可精确表示的 worker/node target；GPU/process 只在 PP=TP=1 退化情况可等价表示，非退化 PP/TP 下明示 fail-close，不将单 shard 错扩大为整个 replica；
- 外部项目只作为实现参考或可替换 latency/network backend，不能改变 pinned SGLang 的请求调度控制流。

## 4. 证据边界

开源项目的存在只证明可参考的建模方法，不证明本项目已经实现、校准或验证了相同能力。当前证据必须分层阅读：

- [`P0 v5`](results/20260829T121349Z_global_des_p0_contract_v5/P0/) 是 12/12 PASS 的 GlobalDES contract lineage；其 summary 明示 `contract_execution_only=true`、`full_scheduler_integration=false`、`p1_p12_execution_ready=false`、`performance_result=false`、`real_hardware_measurement=false`；
- 正式 P1--P12 完成数仍为 **0/2,134**，完整 `12 + 2,134 = 2,146` cells 未运行，因此 `paper_ready=false`；
- 普通 HiCache、AegisKV protection/recovery、Router silent replay 和 exact failure-domain resolution 是当前代码能力，但还没有被正式 P1--P12 performance lineage 覆盖；实现、contract 与论文结果不能互相替代；
- PP2×TP4 只证明 metadata 层的 shard/group contract，不证明 real Scheduler 可以执行任意 PP×TP；非退化拓扑中的单 GPU/process failure 也是 fail-close 边界，不是已实现能力；
- Router 续跑是 synthetic-token metadata continuation，不能写成 logits/RNG-一致的模型续跑；当前 predictor、bandwidth、setup 和 contention 也都是 modeled，不是真机测量；
- P11 workflow-simultaneous coverage 为 0.928571，低于预注册 0.95 gate，且 safe-Slack LCB 非零数为 0；在门未过时不得产生 P11 paper-ready predictor claim。

因此，当前最准确的表述是：**本项目已实现目标的 modeled offline GlobalDES 核心路径，但尚未运行正式实验，也尚未建立真机校准或 paper-ready 性能证据。**
