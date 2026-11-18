# AegisKV Related Works

本目录是 AegisKV 的论文库与建模索引。2026-08-21 的递归审计共覆盖
69 份 PDF、1,169 个物理页和 67 个版本去重后的 canonical works：顶层年份目录
39 份，`kvcache_backup/` 30 份。完整审计见
[`audit/corpus_file_audit.json`](audit/corpus_file_audit.json) 与
[`BASELINE_SELECTION_2026-08-21.md`](BASELINE_SELECTION_2026-08-21.md)。

## 年份与状态规则

- 已正式发表或已接收的论文按 venue year 归档。
- 仅有 preprint 的论文按首次公开年份归档。
- 匿名投稿、保密稿或无法核验公开版本的本地稿件放在
  `2026/unpublished/`；这些文件不能被表述为已发表论文。
- 同一工作的本地匿名稿和后来公开版本可以同时保留，但会明确标注版本。
- `HEXGEN-TEXT2SQL` 固定保留本地 arXiv v1；当前 arXiv v3 已改题并增加作者，
  不用新版本 metadata 覆盖旧 PDF。

完整的逐篇状态、来源、主题和本地路径见 [catalog.tsv](catalog.tsv)，可复用 BibTeX 见
[references.bib](references.bib)。当前 AegisKV 模型只以
[最新设计规范](../design/current_design.md) 为准；早期、范围更宽的 related-work 建模推演已移入
`Archived/2026-08-06-pre-kvcache-centric/research_iterations/`。

## 目录概览

| 目录 | 数量 | 主要内容 |
|---|---:|---|
| `2021/` | 1 | 动态资源划分与 DRL |
| `2024/` | 2 | KV streaming 容错、模块化 prefix reuse |
| `2025/` | 16 | agentic KV、Critical/remaining-path、resilient serving、在线调度 |
| `2026/` | 14 | agent-native serving、故障恢复、state-preserving inference |
| `2026/unpublished/` | 6 | 本地匿名、保密或尚无公开版本的稿件 |
| `kvcache_backup/` | 30 | KV tiering、restoration、reuse、compression、direct GhostServe baseline |

## 对 AegisKV 最直接的论文

### Critical Agent / agentic workflow

- **CAIR (EMNLP 2025)**：用反事实重跑衡量某个 agent activation 对最终输出与
  workflow 轨迹的因果影响；这是“功能关键性”，不能直接替代硬件故障概率。
- **Sherlock (2025)**：用节点故障注入、下游失效范围和回滚/重算量建模 workflow
  vulnerability；fault 仍是语义/执行错误，不是 KV/worker fail-stop。
- **Agentix (NSDI 2026)**：把整个 agent program 作为一等对象；ATLAS 用动态
  DAG 的最长已观测 critical path，而非固定 agent role。
- **Pythia (2026)**：从 workflow/agent trace 学习 dominant execution graph、
  remaining execution 和未来 burst；说明在线模型必须携带置信度和 fallback。
- **KVFlow (2025)**：Agent Step Graph 将未来复用距离传播到 radix/KV node；
  说明 placement 粒度应是共享前缀去重后的 segment，而不是整个 agent。
- **Cortex (2025)**：用 remaining SLO slack、stage selectivity 和 service time 做
  workflow-aware priority；可支持以 Slack 表达工作流延迟传播敏感性，但不能直接证明
  AegisKV 的容错收益模型。
- **Continuum (2025)**：用“未来返回概率乘收益减内存机会成本”选择 TTL；AegisKV
  只借鉴 benefit-versus-opportunity 的比较方式，保持独立建模。

### Inference fault tolerance

- **DejaVu (ICML 2024)**：连续 KV streaming、checkpoint ACK 与恢复边界。
- **FailSafe (2025)**：proactive KV backup、故障后的 load-aware routing 与弹性 TP。
- **LUMEN (2026)**：把恢复拆成故障前 checkpoint placement、故障时 recovery
  dispatch、模型 reload 期间 capacity restoration 三个决策点。
- **Concordia (2026)**：把 KV、scheduler、communication 等 GPU-resident state
  纳入可提交的增量 checkpoint / recovery log。
- **Tarragon (2026)**：将失败位置显式写成 token/layer frontier，并区分有状态
  attention worker 与无状态 expert worker 的恢复成本。
- **GhostServe (MLSys 2026)**：加入 erasure-coded host parity；恢复代价是 GPU/host
  restore、parity reconstruct 与 suffix recompute 的可行下包络。
- **AnchorTP / ReviveMoE / KevlarFlow (2026)**：说明 state-preserving recovery
  不只恢复 KV bytes，还要处理权重、并行组、路由与 serving capacity。

## 当前设计吸收的结论

当前论文范围只吸收以下、与最新设计一致的结论：

1. Algorithm 1 使用当前已揭示的 workflow DAG 计算请求级 causal Slack，不使用静态
   `critical_agent` 标签或额外 critical score；
2. Algorithm 2 针对单 GPU failure，以完整 logical KV chunk 为保护对象；对 Router 固定的
   Peer-HBM target 与共享 Store DRAM 分别执行独立的 `SlackLandlord` 准入，允许
   `HBM_ONLY`、`DRAM_ONLY`、`HBM_AND_DRAM` 或两层均未接纳。两条路径都从仍存活的
   Source Worker 生产 buffers 直接复制，分别使用 D2D/P2P 与 D2H；不存在 HBM 到 DRAM
   的 destage、前置依赖或责任接管。每层只有版本兼容的 all-shard ACK 才能把连续 prefix
   从 `PENDING` 推进为 `COMMITTED`；
3. Algorithm 3 只发布每个 Router-fixed Peer-HBM target 的版本化 HBM
   admission/reservation envelope `alpha`，并根据本地 pressure 运行 guarded AIMD。它不管理
   DRAM、不定义 `beta`、不选择 chunk，也不拥有传输 token；HBM quota 收缩时可以直接删除
   `HBM_ONLY` 或 `HBM_AND_DRAM` 的低价值安全 suffix，缺失的区间在故障时回退到独立
   committed DRAM 或重算；
4. Router 在准入时冻结 `e(r)` 与唯一完整备份/恢复 DP `b(r)`；`e(r)` 的单 GPU
   fail-stop 后整请求重路由到 `b(r)`，不做 stage-level splicing；
5. 目标同时考虑 failure-induced workflow WCT 增量、尾部风险与保护成本，并用 normal-path
   slowdown 硬约束保护健康 workflow；
6. `OPTIMAL` 只表示求解器对已声明有限因果窗口的证明，不表示未知未来上的全局最优。

## 使用注意

- `unpublished/` 中的 Ceres 标有 `Confidential Draft - Do NOT Distribute`，只作为
  当前项目内部建模参考，不应上传、外发或按正式出版物引用。
- HotStorage 匿名稿、Grape 和 Weaver 也只能标为本地稿件/投稿状态。
- `SAGA` 保留为 arXiv-only 邻近工作；其流传的 venue/DOI 信息与官方 program、
  Crossref 核验不一致，不能写成已接收或已发表。
- `FailSafe` 与 MLSys 2026 终稿名 `RaidServe` 是同一工作，本目录不重复存两份。
- `ReaLM` 研究的是 accelerator bit fault / ABFT，属于外围容错工作，不是 serving
  crash 或 KV-state recovery 的直接 baseline。
- AegisKV 与 One Pool、Continuum 是相互独立的问题模型；这里只共享符号组织或
  benefit/cost 比较范式，不写成继承或迁移关系。
