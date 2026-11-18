# MoonKVSim：基于 vLLM 与 Mooncake 的全局虚拟时钟 KV 模拟器实施计划

> 文档状态：规划草案
> 目标项目：`MoonKVSim`
> 推荐仓库路径：`/home/comp/zhanzhao/MoonKVSim`
> 基线分支：`main`
> AegisKV 分支：`aegiskv`
> 时间模型：单一全局虚拟时钟、离线离散事件模拟（DES）
> 更新时间：2026-08-29

## 1. 执行摘要

MoonKVSim 是一个新的、独立于 SGLSim 的纯元数据离散事件模拟器。它以固定版本的 vLLM 为推理引擎语义来源，以固定版本的 Mooncake Store 和 Transfer Engine 为 KV 存储与传输语义来源，以 AIConfigurator 预测 GPU forward 时间，并用一个全局虚拟时钟统一推进所有 DP rank、GPU、Store、网络、故障和恢复事件。

项目分两步实施：

1. 在 `main` 分支实现并验证 Vanilla Mooncake baseline，包括 vLLM-style continuous batching、GPU KV block 生命周期、Mooncake 全局 Store、跨 DP rank KV 复用、DRAM/SSD 容量和传输竞争。
2. 在 `main` 达到基线验收标准并发布 `v0.1.0-mooncake-baseline` 后，从该 tag 创建 `aegiskv` 分支，在同一个 KV 目录和物理数据平面上实现 AegisKV unified KV management、Peer-HBM、SlackLandlord、故障 reroute 和恢复。

最重要的架构边界是：

- `main` 不包含任何 AegisKV 策略，以保证 Mooncake baseline 可独立验证。
- `aegiskv` 不建立第二套备份目录；cache reuse 和 failure protection 必须共享唯一 KV 真相源。
- 不建立持久的 vLLM 私有 Host L2。Host staging buffer 只是传输临时空间；由 Store 管理的本机 DRAM 属于全局 Store DRAM pool。
- AIConfigurator 只预测 GPU forward；Store、网络、RPC、SSD 和恢复时间由独立资源模型产生。
- 设计同步、代码实现、单元测试、模拟结果、真机校准和论文证据必须分别报告。

## 2. 项目目标

### 2.1 功能目标

MoonKVSim 最终需要支持：

- 多节点、多 GPU、多 DP/TP/PP rank 的 vLLM-style 推理部署；
- continuous batching、chunked prefill、decode、preemption 和 recompute；
- GPU HBM 中按 block/page 管理的 active KV 和 prefix KV；
- Mooncake Store 的全局 key、replica、segment、DRAM/SSD 和 eviction；
- 任意兼容 DP rank 通过 block hash 查询和复用已提交 Store KV；
- GPU compute、GPU copy engine、PCIe、NVLink、NIC、fabric、Store DRAM 和 SSD 的竞争；
- process、GPU、node、Store node、Master 和 network failure；
- Router fence、reroute、Store load、recompute 和恢复；
- AegisKV Peer-HBM 与 Store DRAM 独立准入；
- request、failure、deployment 三层 JSON 配置；
- 完整 event journal、容量账本、KV residency timeline、迁移记录和请求指标；
- matched-seed baseline/AegisKV 实验与论文图像。

### 2.2 科学目标

MoonKVSim 应能回答以下问题：

1. Vanilla Mooncake 的跨 DP Store reuse 能在多大程度上降低重算和 TTFT？
2. 故障发生在请求不同阶段时，默认 Mooncake Store、Peer-HBM 和 AegisKV 的恢复收益分别是多少？
3. AegisKV 是否能在相同 HBM/DRAM/网络预算下改善 agent workflow completion time、goodput 和 tail recovery latency？
4. cache reuse value 与 failure protection value 统一管理是否优于两套独立 manager？
5. Peer-HBM budget、Store DRAM capacity、网络带宽和 failure rate 改变时，策略是否仍然稳定？

### 2.3 非目标

首个版本不尝试：

- 执行真实 CUDA、Triton、NCCL 或 RDMA；
- 复刻 vLLM/Mooncake 的每一行实现或线程级 race；
- 用模拟器单独证明真实硬件性能；
- 在 `main` 中提前加入 AegisKV 特殊状态；
- 把 Host staging buffer 误称为持久 Host KV cache；
- 把 Mooncake Master 建模成 KV 字节的数据转发路径；
- 把静态测试通过当作 simulator accuracy 或论文证据。

## 3. 名称、仓库与分支策略

### 3.1 统一命名

| 对象 | 名称 |
|---|---|
| 项目名 | MoonKVSim |
| Git 仓库 | `MoonKVSim` 或 `moonkv-sim` |
| Python package | `moonkvsim` |
| CLI | `moonkv-sim` |
| 配置 schema 前缀 | `moonkvsim.*` |
| 结果 schema 前缀 | `moonkvsim.result.*` |

### 3.2 分支

#### `main`

`main` 只表达 Vanilla Mooncake baseline：

- vLLM engine replicas；
- Router 与 DP selection；
- MooncakeStoreConnector-style shared Store；
- Store Master、Store Client、DRAM/SSD segments；
- 默认 allocation/eviction；
- 默认 failure 行为；
- AIConfigurator 与全局 DES。

#### `aegiskv`

必须在 `main` 验收并打 tag 后创建：

```bash
git switch main
git tag v0.1.0-mooncake-baseline
git switch -c aegiskv v0.1.0-mooncake-baseline
```

`aegiskv` 增加：

- unified reuse/protection KV metadata；
- Peer-HBM pool；
- independent HBM/DRAM admission；
- SlackLandlord；
- HBMQuotaAIMD；
- critical-aware protection；
- failure fence、route version、recovery 和 silent replay；
- AegisKV 实验与图像。

### 3.3 修复同步规则

- 通用 DES、vLLM、Mooncake、predictor 和 artifact bugfix：先进入 `main`，再同步到 `aegiskv`。
- AegisKV-only 策略和 schema：只进入 `aegiskv`。
- baseline 与 treatment 运行必须使用相同 workload、deployment、failure schedule、seed 和 predictor fingerprint。
- 不允许在 `aegiskv` 中悄悄修改 baseline scheduler 或 forward model来改善结果。

## 4. 源码权威与版本锁定

### 4.1 权威来源

| 子系统 | 权威来源 |
|---|---|
| vLLM scheduler、block manager、connector 生命周期 | 固定 commit 的 vLLM 源码 |
| Mooncake Master、Store、Transfer、replica | `/home/comp/zhanzhao/Mooncake` 的固定 baseline commit |
| AegisKV 当前独立层设计 | `/home/comp/zhanzhao/Mooncake/aegiskv/design/current_design.md` |
| SGLSim DES 方法参考 | `/home/comp/zhanzhao/sglsim/python/sglsim/srt/simulation/global_des.py` |
| AIConfigurator adapter 参考 | `/home/comp/zhanzhao/sglsim/python/sglsim/srt/time_predictor/` |

### 4.2 `source-lock.json`

新仓库必须首先创建：

```text
reference/source-lock.json
```

建议 schema：

```json
{
  "schema": "moonkvsim.source-lock.v1",
  "vllm": {
    "repository": "",
    "commit": "",
    "version": "",
    "patches": []
  },
  "mooncake": {
    "repository": "/home/comp/zhanzhao/Mooncake",
    "baseline_commit": "",
    "aegiskv_commit": ""
  },
  "sglsim_reference": {
    "repository": "/home/comp/zhanzhao/sglsim",
    "commit": ""
  },
  "aiconfigurator": {
    "package_version": "",
    "database_mode": "",
    "database_sha256": "",
    "hardware_profile": ""
  }
}
```

### 4.3 源码对齐记录

每个模拟模块必须在 `reference/` 中记录：

- 对应源码文件和函数；
- 被保留的控制流；
- 被抽象掉的细节；
- 有意的模拟替换点；
- 已知 fidelity gap；
- 验证 trace 和结果。

## 5. 总体架构

```text
Workload / Agent DAG / Sessions
                │
                ▼
        Global Request Router
                │
       route / reroute / fence
                │
     ┌──────────┼──────────┐
     ▼          ▼          ▼
 vLLM DP0    vLLM DP1   vLLM DP2
 Scheduler   Scheduler  Scheduler
 GPU HBM     GPU HBM    GPU HBM
     │          │          │
     └──────────┼──────────┘
                ▼
      Mooncake Connector Model
                │
                ▼
       Global KV Directory
                │
      ┌─────────┴──────────┐
      ▼                    ▼
 Store Master         Transfer Engine
 allocation           GPU/DRAM/SSD copy
 replica metadata     resource reservation
      │
      ▼
 Store Nodes
 ├── DRAM segments
 └── SSD segments
```

在 `aegiskv` 分支增加：

```text
Global KV Directory
├── PRIMARY_HBM
├── PEER_HBM       AegisKV protection pool
├── STORE_DRAM     shared reuse + protection pool
└── STORE_SSD      optional Store fallback
```

## 6. 唯一 KV 真相源

### 6.1 KV identity

每个可共享 KV object/block/chunk 至少包含：

- tenant ID；
- model ID 与 revision；
- KV dtype；
- KV ABI/layout；
- TP/PP layout；
- block size；
- stable block hash；
- token interval；
- shard identity；
- request/session/workflow lineage；
- route version；
- source failure domain。

### 6.2 KV location

建议位置类型：

```text
PRIMARY_HBM
PEER_HBM
STORE_DRAM
STORE_SSD
```

`main` 只使用 `PRIMARY_HBM`、`STORE_DRAM` 和 `STORE_SSD`；`PEER_HBM` 只在 `aegiskv` 出现。

### 6.3 Replica state

状态名称最终应对齐 pinned Mooncake，但至少必须表达：

```text
ALLOCATED
PENDING
COMMITTED
ABORTED
LOST
EVICTED
REMOVED
```

硬约束：

- `PENDING` 不可用于 cache hit 或 recovery；
- 只有所有 shards 完成且版本兼容才可 `COMMITTED`；
- failure/eviction 必须原子地更新目录与容量账本；
- Store Master 只管理逻辑位置和状态，不搬运 KV bytes；
- 一个物理副本只能计费一次。

### 6.4 Unified role

`aegiskv` 分支中的一个副本可以同时具有多个 role：

```json
{
  "location": "STORE_DRAM",
  "roles": ["CACHE_REUSE", "FAILURE_PROTECTION"],
  "state": "COMMITTED"
}
```

这样普通 cache lookup 和故障恢复查询相同目录，不再维护 HiCache/AegisKV 两套真相源。

## 7. DRAM 与 L2 语义

MoonKVSim baseline 不定义持久、rank-private Host L2。

### 7.1 Host staging

- 只用于无 GDR/TCP 等传输路径的临时 buffer；
- 生命周期与 transfer operation 绑定；
- 不参与 prefix lookup；
- 不提供跨请求复用；
- 不计入 Store DRAM residency。

### 7.2 Store DRAM

- 由 Mooncake Store Client/daemon 提供；
- 由 Master 统一分配和发布；
- 对所有兼容 DP rank 全局可见；
- 本机与远程 segment 属于同一个逻辑 pool；
- locality 只影响复制路径、带宽、setup 和 failure correlation；
- GPU/进程软件故障后，只要 Store failure domain 存活，副本仍可使用。

## 8. 全局虚拟时钟 DES

### 8.1 基本模型

模拟器使用单一、单调递增的 `now_ns`：

```python
schedule_at(timestamp_ns, event)
schedule_after(delay_ns, event)
reserve(resources, ready_ns, duration_ns)
run(until_ns=None)
```

所有事件按以下 key 排序：

```text
(timestamp_ns, priority, stable_sequence)
```

禁止：

- `sleep()`；
- rank-local clock 最后再取 max；
- transfer 完成前提前发布 metadata；
- 在一个 aggregate delay 中隐藏多个资源竞争；
- 在失败后继续提交已失效 source/target 的传输。

### 8.2 事件类别

```text
REQUEST_ARRIVAL
ROUTE_REQUEST
SCHEDULER_STEP_READY
FORWARD_START
FORWARD_COMPLETE
KV_BLOCK_ALLOCATE
KV_BLOCK_FREE
STORE_QUERY
PUT_START
PUT_ALLOCATED
TRANSFER_START
TRANSFER_COMPLETE
PUT_COMMIT
GET_START
GET_COMPLETE
EVICTION_START
EVICTION_COMPLETE
FAILURE_INJECT
FAILURE_DETECT
ROUTE_FENCE
REROUTE
RECOVERY_START
RECOVERY_COMPLETE
SILENT_REPLAY_COMPLETE
REQUEST_COMPLETE
REPAIR_COMPLETE
```

### 8.3 同时间事件顺序

必须在设计阶段冻结并测试：

- failure 与 transfer completion 同时发生时谁先执行；
- fence 与 output token 同时发生时是否允许输出；
- commit 与 eviction 同时发生时副本是否可见；
- request completion 与保护副本释放同时发生时如何计费；
- repair 与 reroute 同时发生时哪些 route version 合法。

默认建议采用保守故障语义：在相同 timestamp 上先处理 failure/fence，再处理尚未完成的 commit/output；最终仍以固定源码和实验协议为准。

## 9. 物理资源模型

### 9.1 GPU

每张 GPU 独立维护：

- weights；
- runtime reserve；
- primary KV；
- prefix/reusable KV；
- Peer-HBM（仅 `aegiskv`）；
- recovery pending；
- free HBM；
- compute resource；
- D2D/H2D/D2H copy engines。

### 9.2 节点与网络

- PCIe ingress/egress；
- NVLink/NVSwitch；
- NIC ingress/egress；
- RDMA/TCP protocol；
- cross-node fabric；
- NUMA/locality；
- full-duplex/half-duplex 配置；
- setup、bandwidth、queue depth 和 contention policy。

### 9.3 Store

每个 Store node/segment 独立维护：

- media type；
- capacity；
- used/free bytes；
- allocator state；
- DRAM read/write lanes；
- SSD read/write queues；
- health；
- failure domain；
- eviction watermark；
- local placement metadata。

### 9.4 容量守恒

任意时刻必须满足：

```text
capacity = fixed_reserve + pending + committed + free
```

并且：

- pending transfer 先 reserve，再传输；
- abort/lost/evict 必须释放容量；
- external alias 不得重复计费；
- 一个 shard 不能同时被两个 internal account拥有；
- artifact 中的 occupancy 必须可由事件日志重建。

## 10. vLLM-style 推理引擎

### 10.1 必须对齐的行为

- request/sequence 生命周期；
- waiting、running、preempted、finished；
- continuous batching；
- chunked prefill；
- decode allocation；
- paged KV block；
- prefix block hash；
- cached block lookup；
- preemption 与 recompute；
- block free；
- scheduler output；
- connector metadata；
- DP routing hints。

### 10.2 DP 与 Router 边界

- 外部 Router 选择初始 DP target；
- vLLM 内部 DP balancing 与外部 routing 分开记录；
- Mooncake Store 提供 KV lookup，不决定最终 target；
- cache-aware Router 可以使用目录 hit 信息，但必须记录评分和选择原因；
- reroute 必须生成新 route version并 fence旧执行。

### 10.3 Prefix key 一致性

跨 DP cache reuse 至少要求：

- 一致的 `PYTHONHASHSEED`；
- 一致的 block size；
- 一致 model/revision；
- 一致 KV ABI/layout；
- 一致 tenant/namespace；
- 一致 TP/PP layout 或明确 reshard 路径。

## 11. AIConfigurator 预测器

### 11.1 Engine-neutral 输入

建议公共输入：

```python
PredictionRequest(
    extend_length: int,
    past_kv_length: int,
)

PredictionBatch(
    requests: list[PredictionRequest],
    phase: Literal["prefill", "decode"],
)
```

vLLM adapter 负责把实际 scheduled batch 转换为预测输入。

### 11.2 预测边界

AIConfigurator 只返回：

```text
predicted_forward_duration_ns
```

以下时间必须独立建模：

- scheduler CPU；
- connector metadata；
- Master RPC；
- D2D/H2D/D2H；
- RDMA/TCP；
- Store DRAM/SSD；
- failure detection；
- reroute；
- recovery replay。

### 11.3 校准

- prefill 和 decode 分开；
- 不直接继承 SGLSim HiCache scale；
- 不使用一个全局 scale factor吸收 shape error；
- 对 batch size、input length、past KV length 做 shape-aware 验证；
- 保存 raw prediction、calibrated prediction、observed value 和 residual；
- 未完成 vLLM 真机校准前，结果标记为 modeled/preliminary。

## 12. Vanilla Mooncake 模型（`main`）

### 12.1 Master

模拟：

- segment mount/unmount；
- client registration；
- key/object metadata；
- replica list；
- allocation strategy；
- `PutStart`；
- finalize/commit；
- `GetReplicaList`；
- lease；
- remove；
- tenant quota；
- high-watermark eviction trigger；
- node join/leave；
- Master failure/repair（后期）。

### 12.2 Store Node

模拟：

- local DRAM allocation；
- SSD allocation；
- fragmentation/granularity；
- local eviction候选；
- watermarks；
- read/write queue；
- node failure；
- replica lost notification。

### 12.3 Transfer Engine

模拟实际数据路径而非一个总延迟：

```text
GPU → Store DRAM
Store DRAM → GPU
GPU → temporary host staging → TCP → Store
Store SSD → Store DRAM → GPU
GPU → remote GPU（如基线 connector 需要）
```

每条路径必须声明：

- source/destination；
- byte count；
- setup；
- required resources；
- bottleneck bandwidth；
- ready/start/end；
- state；
- abort reason。

### 12.4 Baseline recovery

Vanilla Mooncake 也可能已经存在 committed Store KV。因此默认故障路径应是：

```text
GPU/process failure
→ Router reroute
→ query Store by compatible block hash
→ load committed continuous prefix
→ recompute missing suffix
```

AegisKV 的收益不能错误地与“全部重算且完全没有 Mooncake Store”的弱 baseline 比较。

## 13. AegisKV 设计修订（`aegiskv`）

在实现前，必须把当前设计从 protection overlay 修订为 unified KV management。

### 13.1 两个策略 pool

- Peer-HBM pool；
- global Store DRAM pool。

Store DRAM 中的 local/remote segments 不是两个独立算法 tier，只是不同 placement、path 和 failure domain。

### 13.2 独立准入

同一 sealed chunk 的 HBM 和 DRAM 决策必须独立支持：

```text
HBM_ONLY
DRAM_ONLY
HBM_AND_DRAM
BYPASS
```

两条路径：

- Source production HBM → designated Peer-HBM；
- Source production HBM → Store DRAM final space。

禁止：

- Peer-HBM → Store DRAM writeback；
- target-owned background destage；
- DRAM 成功作为 HBM 准入前提；
- 一条路径失败终止另一条路径。

### 13.3 Unified value

建议策略价值由两部分构成：

```text
total_value
= expected future reuse saving
+ Slack-weighted recovery harm avoided
- transfer cost
- storage rent
```

请求执行期间副本可以同时具有 reuse 和 recovery value；请求完成后 recovery value 归零，但 reuse value 可以继续保留。

### 13.4 HBMQuotaAIMD

- 只调整 Peer-HBM quota；
- DRAM 由 Store capacity和策略管理；
- HBM contraction 停止新 admission并淘汰低价值安全尾部；
- 删除 HBM 不要求 DRAM 已提交；
- 无其他副本时允许以后重算。

## 14. 故障与恢复模型

### 14.1 故障类型

```text
PROCESS_FAILURE
GPU_SOFTWARE_FAILURE
GPU_HARD_FAILURE
INFERENCE_NODE_FAILURE
STORE_NODE_FAILURE
MASTER_FAILURE
NIC_FAILURE
NETWORK_PARTITION
CORRELATED_RACK_FAILURE
```

### 14.2 failure domain

每个 location 和 resource 都要绑定 failure domain：

- process；
- GPU；
- inference node；
- Store process；
- Store node；
- NIC；
- rack/fabric；
- Master replica。

必须显式定义：

- GPU software failure 是否保留独立 Store daemon 的本机 DRAM；
- node failure 是否同时丢失 GPU、host staging 和本机 Store segment；
- network partition 是否造成 temporary unavailable而非 replica lost；
- repair 后旧 replica 是否重新可见。

### 14.3 恢复协议

```text
failure occurs
→ detection
→ fence source and output
→ mark failed-domain locations unavailable/lost
→ select healthy compatible target
→ increment route version
→ query committed continuous prefix
→ choose Peer-HBM / Store DRAM / Store SSD / recompute
→ reserve destination HBM and transfer resources
→ load all required shards
→ materialize target block table
→ silent replay to committed output position
→ atomic activation
→ resume output
```

硬约束：

- 未提交副本不可恢复；
- 不兼容 model/revision/ABI/layout 不可恢复；
- 只采用完整、连续 prefix；
- 所有 TP/PP shards完成后才能激活；
- stale route 不可输出；
- output token 不可重复或丢失；
- recovery adoption 后再释放旧保护容量。

## 15. 三层 JSON 配置

### 15.1 `requests.json`

负责请求层：

```json
{
  "schema": "moonkvsim.requests.v1",
  "seed": 1,
  "arrival": {
    "mode": "trace",
    "request_rate": null,
    "max_concurrency": 128
  },
  "workload": {
    "kind": "agent_workflow",
    "trace_path": "",
    "prefix_sharing": true,
    "closed_loop": true
  },
  "defaults": {
    "input_tokens": 4096,
    "output_tokens": 512
  }
}
```

必须记录：

- trace和hash；
- arrival mode；
- request rate；
- concurrency；
- prompt/output；
- session/workflow；
- tool delay；
- shared prefix；
- seed。

### 15.2 `failure.json`

负责故障注入层：

```json
{
  "schema": "moonkvsim.failure.v1",
  "enabled": true,
  "events": [
    {
      "failure_id": "f0",
      "type": "GPU_SOFTWARE_FAILURE",
      "at_s": 30.0,
      "target": {"node_id": "n0", "gpu_id": 0},
      "detection_delay_ms": 100,
      "repair_after_s": null
    }
  ]
}
```

必须记录：

- failure type；
- exact target或随机选择规则；
- injection time/distribution；
- detection；
- repair；
- correlated scope；
- network partition方向；
- reroute规则。

### 15.3 `deployment.json`

负责部署层：

```json
{
  "schema": "moonkvsim.deployment.v1",
  "model": {
    "id": "Qwen/Qwen3-8B",
    "revision": "",
    "dtype": "bf16",
    "kv_dtype": "bf16"
  },
  "engine": {
    "kind": "vllm",
    "source_commit": "",
    "dp_size": 8,
    "tp_size": 1,
    "pp_size": 1,
    "block_size": 16
  },
  "store": {
    "kind": "mooncake",
    "dram_segments": [],
    "ssd_segments": [],
    "eviction_policy": "baseline"
  },
  "network": {
    "protocol": "rdma",
    "gdr": true,
    "links": []
  },
  "predictor": {
    "kind": "aiconfigurator",
    "profile": ""
  },
  "policy": {
    "kind": "mooncake_default"
  }
}
```

`aegiskv` 分支将 `policy.kind` 改为 `aegiskv` 并加入显式 HBM/DRAM policy 配置。

### 15.4 Run manifest

每次运行生成：

```text
resolved-config.json
manifest.json
SHA256SUMS
```

manifest 必须包括：

- 三个配置文件及 hash；
- code commit和dirty状态；
- source-lock hash；
- predictor database hash；
- workload trace hash；
- seed；
- start/end virtual time；
- validation/evidence class；
- result completeness。

## 16. 输出与可观测性

### 16.1 规范化输出

```text
event-journal.jsonl
request-metrics.parquet
batch-events.parquet
kv-residency-events.parquet
kv-transfer-events.parquet
resource-utilization.parquet
store-directory-snapshots.parquet
failure-events.parquet
recovery-events.parquet
capacity-audit.json
run-audit.json
summary.json
```

### 16.2 必须可重建的状态

- 每个 DP rank waiting/running/preempted 队列；
- 每张 GPU 的 HBM账户；
- 每个 Store segment 的 DRAM/SSD 占用；
- 每个 KV block/chunk 的 locations、roles和state；
- 每次 KV 迁移的 source、destination、bytes、start/end；
- 每个 network/resource 的 busy intervals；
- 每个 failure 的检测、fence、reroute和恢复过程；
- 每个请求 TTFT、TPOT、E2E、recompute、restore和WCT贡献。

## 17. 仓库结构

```text
MoonKVSim/
├── README.md
├── DESIGN.md
├── pyproject.toml
├── reference/
│   ├── source-lock.json
│   ├── vllm-semantics.md
│   ├── mooncake-semantics.md
│   ├── aiconfigurator-notes.md
│   └── alignment-notes.md
├── schemas/
│   ├── requests.schema.json
│   ├── failure.schema.json
│   └── deployment.schema.json
├── python/moonkvsim/
│   ├── des/
│   ├── engine/vllm/
│   ├── store/mooncake/
│   ├── predictor/
│   ├── topology/
│   ├── router/
│   ├── failure/
│   ├── policy/
│   ├── metrics/
│   └── cli.py
├── experiments/
│   ├── configs/
│   ├── scripts/
│   ├── results/
│   └── figures/
└── tests/
    ├── unit/
    ├── integration/
    ├── differential/
    └── golden/
```

`aegiskv` 分支增加：

```text
python/moonkvsim/policy/aegiskv/
├── controller.py
├── slack_landlord.py
├── hbm_quota_aimd.py
├── recovery_planner.py
├── failure_fence.py
└── audit.py
```

## 18. 分阶段实施与验收

### M0：项目冻结与初始化

任务：

- [ ] 创建新仓库；
- [ ] 写 README/DESIGN；
- [ ] 固定 vLLM/Mooncake/SGLSim/AIConfigurator版本；
- [ ] 创建三层 JSON schema；
- [ ] 建立 CI、lint、test骨架；
- [ ] 建立 evidence class 和 manifest规范。

验收：

- source lock完整；
- 所有架构术语无歧义；
- baseline和AegisKV边界明确；
- 未写任何未经源码核对的调度逻辑。

### M1：通用 DES kernel

任务：

- [ ] stable event heap；
- [ ] single `now_ns`；
- [ ] cancellation；
- [ ] resource calendars；
- [ ] event journal；
- [ ] deterministic replay；
- [ ] invariant audit。

验收：

- 时间不倒退；
- 相同 seed byte-identical结果；
- 资源无非法重叠；
- failure能够取消传输且释放reservation；
- journal可重建状态。

### M2：vLLM engine baseline

任务：

- [ ] request states；
- [ ] block manager；
- [ ] continuous batching；
- [ ] chunked prefill；
- [ ] decode；
- [ ] preemption/recompute；
- [ ] prefix hash；
- [ ] DP router。

验收：

- 小型golden workload与pinned vLLM的batch/token/block序列对齐；
- HBM容量守恒；
- DP replica独立调度但共享全局时钟。

### M3：AIConfigurator

任务：

- [ ] engine-neutral prediction schema；
- [ ] vLLM batch adapter；
- [ ] prefill/decode预测；
- [ ] profile fingerprint；
- [ ] calibration output。

验收：

- predictor不依赖HiCache；
- forward与I/O分开；
- shape-aware误差报告完整；
- 未校准结果明确标记preliminary。

### M4：Mooncake Store baseline

任务：

- [ ] Master；
- [ ] directory；
- [ ] segments；
- [ ] allocator；
- [ ] Store Client；
- [ ] Put/Get/Remove；
- [ ] DRAM/SSD；
- [ ] eviction；
- [ ] Transfer Engine资源路径。

验收：

- DP0写入后DP1可query/load；
- pending不可见；
- committed才可复用；
- Master不搬字节；
- Store容量守恒；
- local/remote segment路径正确。

### M5：Baseline failure与端到端验证

任务：

- [ ] process/GPU/node/Store/network failure；
- [ ] fence与reroute；
- [ ] Store prefix load；
- [ ] missing suffix recompute；
- [ ] small physical validation；
- [ ] baseline experiment smoke。

验收：

- reroute后无重复输出；
- committed Store KV可跨DP恢复；
- failed-domain副本不可用；
- results可复现；
- 发布`v0.1.0-mooncake-baseline`。

### M6：创建 `aegiskv` 分支并冻结新设计

任务：

- [ ] 从baseline tag创建分支；
- [ ] 修订AegisKV unified manager设计；
- [ ] 冻结KV role/state/schema；
- [ ] 定义reuse+recovery value；
- [ ] 定义Peer-HBM和Store DRAM边界；
- [ ] 更新算法、配置、测试和artifact协议。

验收：

- 设计无HiCache overlay语言；
- DRAM明确等于global Store DRAM；
- 一个副本只计费一次；
- 独立tier语义保持；
- 禁止HBM→DRAM destage。

### M7：AegisKV normal path

任务：

- [ ] sealed chunks；
- [ ] Peer-HBM candidate；
- [ ] Store DRAM candidate；
- [ ] independent admission；
- [ ] PENDING/COMMITTED/ABORTED；
- [ ] SlackLandlord；
- [ ] unified role；
- [ ] HBMQuotaAIMD。

验收：

- HBM_ONLY/DRAM_ONLY/HBM_AND_DRAM/BYPASS均有测试；
- 一个tier失败不终止另一个；
- bytes来自source production buffer；
- Store DRAM副本可被普通cache lookup复用；
- capacity/rent/frontier审计通过。

### M8：AegisKV failure path

任务：

- [ ] route version；
- [ ] failure fence；
- [ ] standby rotation；
- [ ] recovery source选择；
- [ ] Peer-HBM activation/load；
- [ ] Store DRAM load；
- [ ] recompute；
- [ ] silent replay；
- [ ] atomic output activation。

验收：

- stale route不能输出；
- incomplete shards不能恢复；
- failed target的Peer-HBM不能重新标记；
- independent Store DRAM record保留；
- 恢复过程和容量hold完整可审计。

### M9：校准、实验和论文资产

任务：

- [ ] 组件级校准；
- [ ] baseline/AegisKV matched runs；
- [ ] failure matrix；
- [ ] scale/sensitivity；
- [ ] ablation；
- [ ] figure scripts；
- [ ] report与evidence matrix。

验收：

- 所有图由规范化结果自动生成；
- 图中的每个点可追溯到manifest；
- modeled、calibrated、physical evidence分开；
- 没有缺失cell或静默fallback；
- paper-ready gate单独审计。

## 19. 测试计划

### 19.1 Unit tests

- event排序和取消；
- resource reservation；
- HBM/DRAM/SSD账本；
- KV state machine；
- block hash和namespace；
- allocator/eviction；
- predictor adapter；
- failure-domain invalidation；
- route version与fence。

### 19.2 Integration tests

- 单DP推理；
- 多DP并发；
- DP0→Store→DP1复用；
- DRAM capacity pressure；
- SSD promotion；
- process failure reroute；
- node failure；
- network contention；
- pending transfer被故障终止。

### 19.3 Differential tests

- vLLM scheduler trace；
- block allocate/free trace；
- Mooncake API/state sequence；
- default eviction order；
- connector metadata；
- cross-DP prefix hit。

### 19.4 AegisKV tests

- 四种tier组合；
- HBM/DRAM independent failure；
- target-first failure；
- source-first failure；
- simultaneous failure/commit；
- route rotation；
- partial shard completion；
- HBM AIMD contraction；
- unified reuse/protection role；
- request completion后reuse保留。

## 20. 验证与证据分级

### E0：设计

只有文档、schema和源码映射。

### E1：静态实现

代码存在、类型检查和单元测试通过；不能声称运行时或性能正确。

### E2：模拟器语义验证

golden/differential trace对齐；可以声称控制流和状态机模拟已验证。

### E3：组件校准

forward、transfer、Store、network分别有物理测量；可以声称该参数区间内经过校准。

### E4：端到端验证

小规模真实vLLM+Mooncake与MoonKVSim matched；可以报告simulator accuracy。

### E5：论文实验

完整matrix、重复、置信区间、audit和provenance通过；才可作为论文结论。

任何低等级证据不得自动升级为高等级证据。

## 21. 实验矩阵

### 21.1 系统对比

```text
Recompute only
Vanilla Mooncake DRAM
Vanilla Mooncake DRAM+SSD
Static Peer-HBM
AegisKV DRAM only
AegisKV Peer-HBM only
AegisKV HBM+DRAM
Full AegisKV unified manager
```

### 21.2 消融

- 无Slack；
- 无reuse value；
- 无recovery value；
- 两套manager而非unified manager；
- 无AIMD；
- static HBM quota；
- random/LRU/LFU eviction；
- 无failure-domain awareness；
- 无cache-aware routing；
- 不同chunk size。

### 21.3 敏感性

- DP/TP/PP规模；
- 请求率和并发；
- prefix sharing；
- agent workflow长度和criticality；
- failure time/type/rate；
- HBM budget；
- Store DRAM budget；
- SSD capacity；
- RDMA/TCP/GDR；
- NIC/fabric带宽；
- Store node数量；
- detection/reroute delay。

### 21.4 重复性

- matched seed；
- baseline/treatment使用相同请求和failure schedule；
- 至少多个独立seed；
- 报告均值、分位数和置信区间；
- 明确剔除/失败run；
- 禁止只选择成功run。

## 22. 论文图像计划

### F1：系统架构

vLLM replicas、Router、global directory、Store Master、Store nodes、Peer-HBM和统一KV role。

### F2：故障恢复时间线

failure→detection→fence→reroute→load/recompute→silent replay→resume。

### F3：每GPU HBM占用时间线

weights、primary KV、prefix KV、Peer-HBM、recovery pending、free。

### F4：Store DRAM/SSD占用时间线

每个Store node和segment的pending、committed、evicted和free。

### F5：KV迁移Gantt

展示D2D、D2H、RDMA/TCP、H2D、SSD promotion及资源竞争。

### F6：端到端WCT/goodput

baseline与AegisKV matched comparison。

### F7：恢复延迟CDF

按Peer-HBM、Store DRAM、SSD和recompute分组。

### F8：保护覆盖与重算

protected continuous prefix、restore tokens、recompute tokens。

### F9：策略消融

Slack、reuse value、AIMD、unified manager。

### F10：预算敏感性

Peer-HBM/Store DRAM预算与WCT、goodput、恢复收益。

### F11：规模扩展

DP数量、Store nodes、网络带宽与性能。

### F12：模拟器准确性

AIConfigurator、传输和端到端observed-vs-predicted误差。

## 23. 主要风险与应对

### 风险 1：vLLM 源码版本不明确

应对：先获取和固定源码，不在此之前实现调度器。

### 风险 2：直接复用 SGLSim 导致 SGLang 语义泄漏

应对：只抽取通用 DES/predictor schema，不复用 HiCache、ScheduleBatch或tier enum。

### 风险 3：Mooncake baseline 被弱化

应对：baseline必须支持跨DP Store reuse和故障后读取已提交Store KV。

### 风险 4：双重 KV 计费

应对：global directory唯一真相源；physical copy与logical roles分离。

### 风险 5：AIConfigurator 校准不可迁移

应对：对vLLM重新采集prefill/decode校准，不沿用HiCache scale。

### 风险 6：传输时间被aggregate delay隐藏

应对：每个copy显式预约所有物理资源并记录start/end。

### 风险 7：模拟器结果被过早当成论文证据

应对：执行E0-E5证据门槛；paper-ready单独审计。

### 风险 8：`main` 与 `aegiskv` 漂移

应对：baseline tag、通用bugfix先main、matched config和source-lock强制检查。

## 24. Definition of Done

### 24.1 `main` 完成

- [ ] pinned vLLM semantics已对齐；
- [ ] Vanilla Mooncake Store语义已对齐；
- [ ] 跨DP shared KV可用；
- [ ] 无持久私有Host L2；
- [ ] 全局虚拟时钟驱动全部组件；
- [ ] AIConfigurator只驱动forward；
- [ ] 容量和资源竞争严格审计；
- [ ] baseline failure/reroute可运行；
- [ ] 小规模端到端验证完成；
- [ ] 发布`v0.1.0-mooncake-baseline`。

### 24.2 `aegiskv` 完成

- [ ] unified directory是唯一KV真相源；
- [ ] cache reuse和failure protection互相可见；
- [ ] Peer-HBM与Store DRAM独立准入；
- [ ] 四种residency组合完整；
- [ ] 无HBM→DRAM writeback；
- [ ] failure/fence/reroute/recovery/silent replay闭环；
- [ ] 每个DP/GPU/Store/transfer状态可观测；
- [ ] matched baseline/AegisKV实验完成；
- [ ] calibration和端到端验证通过；
- [ ] 完整实验matrix和图像通过provenance audit。

## 25. 立即执行顺序

1. 创建 `/home/comp/zhanzhao/MoonKVSim` 空仓库。
2. 创建 `README.md`、`DESIGN.md` 和本计划副本。
3. 获取并固定 vLLM 源码版本。
4. 确定 Mooncake vanilla baseline commit 与 AegisKV commit。
5. 生成 `reference/source-lock.json`。
6. 写三层 JSON schema 和 resolved manifest schema。
7. 实现独立 DES kernel，不迁移任何 SGLang tier。
8. 用 toy workload 验证虚拟时钟和资源预约。
9. 实现 pinned vLLM scheduler/block manager。
10. 接入 engine-neutral AIConfigurator adapter。
11. 实现 Mooncake Master/Store/Transfer baseline。
12. 完成跨 DP Store reuse 和 baseline failure验证。
13. 发布 baseline tag。
14. 创建 `aegiskv` 分支并先修订设计。
15. 依次实现 normal path、failure path、实验和图像。

---

本计划是 MoonKVSim 的实施规范，不代表任何模块已经实现、校准或验证。后续每个 milestone 应更新独立状态、验收证据和未决问题，不能用计划完成度替代代码与实验完成度。
