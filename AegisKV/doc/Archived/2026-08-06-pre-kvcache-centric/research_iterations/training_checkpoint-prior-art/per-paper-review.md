# 2025–2026 LLM Training Checkpoint / Fault-Tolerance Paper Review

本目录固定收录 30 篇论文。PDF 已逐文件验证 `%PDF-` magic、可解析页数、SHA-256，并确认无重复哈希；下载级证据见 `download_validation.csv`，合并后的机器可读索引见 `manifest.csv`。

## 口径与结论

- 发表口径：main conference **28** 篇，peer-reviewed workshop **1** 篇，work-in-progress **1** 篇。后两类在每篇条目中显式标注，不冒充顶会主会论文。
- 语义分组：运行时 snapshot / state replication **11** 篇；格式、后处理、读取或经验 I/O **4** 篇；只做恢复、诊断、校验或资源规划且不定义新写路径 **15** 篇。
- 对 11 篇“运行时 snapshot / state replication”论文逐一看第一落点：**9/11 明确把 Host DRAM（本机、peer 或独立 CPU server）作为第一备份/暂存层**；PhoenixOS 是可配置 DRAM/SSD/remote-DRAM backend；Di-PS 的状态原生位于 CPU PS，但未披露 snapshot destination。
- 在这 9 篇明确 Host-first 的论文中，**5/9 给出 Host→持久层的明确路径**（PCcheck、FlowCheck、ByteCheckpoint、AdaCheck、RobustRL）；**1/9 只提及 Host+SSD 分层却未给 drain/commit**（ByteRobust）；**3/9 原型停在 Host memory**（Checkmate、MoEvement、GCR）。因此严谨结论是：**Host DRAM 作为快速第一层是主流；Host→disk/remote persistent storage 不是所有系统的共同终点，更不是每篇都给出可验证 commit 语义。**
- 另有 85,000-checkpoint WIP 观察到生产作业常先写 node-local storage，再异步 drain 到 shared storage；它没有披露 node-local 介质，因此不能把该结果自动计入 Host-DRAM-first。
- `async` 必须拆成至少三种：snapshot/create 异步、Host/peer replication 异步、persistent drain 异步；只有最后的 distributed integrity / fence / msync / replica-window completion 才能作为相应层级的 commit。

Host→storage 分类的原始计数（用于复核，不代表质量排序）：

- `configurable_backend_includes_host_and_ssd`: 1
- `cpu_native_source_destination_unspecified`: 1
- `host_only_no_persistent_drain`: 3
- `host_to_persistent_explicit`: 4
- `host_to_persistent_explicit_parallel_paths`: 1
- `host_to_persistent_mentioned_not_specified`: 1
- `local_storage_to_shared_explicit_medium_unspecified`: 1
- `not_applicable_analytical_model`: 1
- `not_applicable_consumes_external_checkpoint`: 2
- `not_applicable_live_snapshot`: 1
- `not_applicable_load_only`: 1
- `not_applicable_no_checkpoint_creation`: 12
- `not_applicable_offline_file_transform`: 1

## 逐篇审读

### TC01 — PCcheck: Persistent Concurrent Checkpointing for ML

- 年份 / venue：2025 / ASPLOS 2025（`main`）
- 范围：`direct_checkpoint`；运行时 snapshot / state replication
- 页面 / SHA-256：17 pages / `7ead7e7b15b89348390c6240567b8098e56ea06f5ebec9e783f8d221dced00a9`
- 链接：[venue](https://doi.org/10.1145/3669940.3707255)；[PDF source](https://fotstrt.github.io/files/2025-pccheck.pdf)；本地 `2025-asplos-pccheck.pdf`
- checkpoint 第一落点：本机 pinned Host DRAM（CPU buffer pool）。
- staging path：GPU 状态按 chunk 经 DMA 拷到 pinned DRAM；DRAM chunk 随即由多线程流水写入 SSD 或 PMEM，可同时保留多个 in-flight checkpoint。
- 最终介质：本机/云持久 SSD，或 PMEM。
- redundancy：没有跨机副本；但地址队列保证最近 N+1 个版本中至少一个已持久化版本不被覆盖。
- async boundary：训练与 D2H、DRAM→storage 重叠；多个 checkpoint 可并发，snapshot 与 persist 还可按 chunk 流水。
- commit semantics：所有 writer 完成后，PMEM 路径逐线程 fence，SSD 路径由主线程 msync；随后以计数器/CAS 更新 latest-checkpoint pointer。创建异步，但可恢复版本要到该持久化与指针发布边界后才成立。
- 核心机制：PCcheck 用并发 checkpoint 数、chunk 数、DRAM buffer 和 writer 数共同调度 PCIe、DRAM 与存储带宽。
- 对 AegisKV 的启示：可把 AegisKV 的 α 解释为 Host 快速层预算、β 解释为持久层 drain 预算；提交状态必须像 PCcheck 一样与“已 D2H”分离。
- 证据页：PDF pp. 5–8, 12–13（Fig. 6–7、§3.3、§4.1、结论）。
- 局限：持久层仍是单机故障域；DRAM 容量、PCIe 与 SSD 带宽会限制并发度；PMEM 设备可获得性有限。

### TC02 — FlowCheck: Decoupling Checkpointing and Training of Large-Scale Models

- 年份 / venue：2025 / EuroSys 2025（`main`）
- 范围：`direct_checkpoint`；运行时 snapshot / state replication
- 页面 / SHA-256：16 pages / `2e342f9f0c260dcc201f7266b4bdd8b3af6b7a99728c36aca2619453ab76cae4`
- 链接：[venue](https://doi.org/10.1145/3689031.3696088)；[PDF source](https://jhc.sjtu.edu.cn/~bjiang/papers/Huang_EuroSys2025_FlowCheck.pdf)；本地 `2025-eurosys-flowcheck.pdf`
- checkpoint 第一落点：独立 CPU checkpoint server 的 Host DRAM，而非训练节点 Host memory。
- staging path：交换机镜像 AllReduce 完整梯度 → checkpoint server NIC/I/OAT DMA → huge-page/MMAP 共享内存 → CPU 上重放 optimizer update；约每小时再把维护的状态发送到远端存储池。
- 最终介质：快速层为 CPU checkpoint server DRAM；长期层为 remote storage pool。
- redundancy：建议每个 DP ring 监控至少 3 个训练节点；重复镜像流提供 2/3-copy 梯度包冗余，用于无重传的镜像链路恢复。
- async boundary：镜像、解析、梯度拼接与 CPU optimizer update 必须在单个训练 iteration 内结束，GPU 不等待；无法由冗余恢复丢包时，下一 iteration 退化为常规 checkpoint。
- commit semantics：内存 checkpoint 在收齐本 iteration 完整梯度并完成 CPU update 后与训练状态一致；远端持久版本只有周期传输完成后才成立。论文未给远端原子发布/commit 协议。
- 核心机制：FlowCheck 不拷 GPU snapshot，而是从既有训练网络流重建 checkpoint，并用流内冗余修复镜像丢包。
- 对 AegisKV 的启示：AegisKV 可借鉴“备份来源可不是 GPU snapshot”：算法 2 可在 Host copy、网络旁路重建、持久落盘之间按资源价格选择。
- 证据页：PDF pp. 2, 5, 8–9, 12–13（Fig. 4、§6.2、§7）。
- 局限：需要可编程/可配置镜像网络、额外 CPU servers 与 lossless-enough 流量；加密流量、FSDP/复杂优化器适配和远端 commit 不完整。

### TC03 — ByteCheckpoint: A Unified Checkpointing System for Large Foundation Model Development

- 年份 / venue：2025 / NSDI 2025（`main`）
- 范围：`direct_checkpoint`；运行时 snapshot / state replication
- 页面 / SHA-256：21 pages / `4879f9bfb147e15ed30f4e2c4696f84c780424b6c32c6320f525935cc7422fd9`
- 链接：[venue](https://www.usenix.org/conference/nsdi25/presentation/wan-borui)；[PDF source](https://www.usenix.org/system/files/nsdi25-wan-borui.pdf)；本地 `2025-nsdi-bytecheckpoint.pdf`
- checkpoint 第一落点：pinned CPU memory 的 ping-pong pool。
- staging path：GPU shard → D2H → pinned CPU pool → 多进程序列化 → /dev/shm 共享内存文件 → 上传到 HDFS/NAS/本地盘；HDFS 大文件以定长 sub-file 并发写后做 metadata concat。
- 最终介质：生产主后端是 HDFS；也支持 NAS、本地盘和 in-memory backend，并有 SSD→HDD 冷却层。
- redundancy：消除并行布局造成的重复写：复制态仅一份、shard 各写一次；底层存储副本策略不属于 ByteCheckpoint 机制。
- async boundary：D2H、序列化、shared-memory dump、上传完全流水；完整性 barrier 也改为 gRPC tree 并异步执行。
- commit semantics：分布式 checkpoint 由多文件组成；全部 rank 文件上传后执行完整性检查/atomic save barrier，才是完整 checkpoint。异步返回与该完整性边界不能混同。
- 核心机制：ByteCheckpoint 用并行无关 byte representation、load-time resharding、全异步 I/O pipeline 与后端专用优化统一 LFM checkpoint。
- 对 AegisKV 的启示：AegisKV 的图应显式画出 D2H-ready 与 distributed-commit 两个事件；α/β 控制的是资源与 drain，不等同于 checkpoint 已可发布。
- 证据页：PDF pp. 4–6, 9–10, 12, 19（Fig. 3/5/10、§4.2、App. B）。
- 局限：生产实现依赖大量框架/存储适配；dataloader state 可达数十 GB；远端 HDFS metadata 与超大规模 barrier 仍需工程优化。

### TC04 — Universal Checkpointing: A Flexible and Efficient Distributed Checkpointing System for Large-Scale DNN Training with Reconfigurable Parallelism

- 年份 / venue：2025 / USENIX ATC 2025（`main`）
- 范围：`direct_checkpoint`；格式、后处理、读取或经验 I/O
- 页面 / SHA-256：17 pages / `c70eedd045aac3d0d51f93842891761996bdfa6d38fb1b1c75803c6f7ad94b0f`
- 链接：[venue](https://www.usenix.org/conference/atc25/presentation/lian)；[PDF source](https://www.usenix.org/system/files/atc25-lian.pdf)；本地 `2025-atc-universal-checkpointing.pdf`
- checkpoint 第一落点：N/A：UCP 接收已经写入持久存储的 distributed checkpoint；它不定义训练时 GPU snapshot 的第一落点。
- staging path：persistent distributed files → CPU 并行 Extract/Union/StripPad → per-parameter atomic checkpoint files → 目标 rank 按 layer 经 CPU memory/H2D 加载。
- 最终介质：atomic checkpoint files 所在持久存储。
- redundancy：通过 redundancy-bypassing loading 去掉 DP 复制读；这是 I/O 去重，不是故障副本。
- async boundary：转换仅在需要重配置时 lazy 触发；不是与训练重叠的 snapshot/persist pipeline。
- commit semantics：Save 生成 per-parameter atomic files；论文强调结构上的 atomic checkpoint，但未给跨文件崩溃一致性或发布事务。
- 核心机制：以 parallelism-independent atomic checkpoint 和 pattern-based reconfiguration 支持 ZeRO/TP/PP/3D 之间恢复。
- 对 AegisKV 的启示：AegisKV 可把恢复布局兼容性作为算法 2 的可行性约束，但不要把格式原子性当作备份 durability。
- 证据页：PDF pp. 5–10, 13–14（Fig. 2、§5.2–§5.4）。
- 局限：依赖已有基础 checkpoint；转换是重配置成本而非快照优化；“atomic”主要指参数粒度表示，不应误读为持久事务提交。

### TC05 — Robust LLM Training Infrastructure at ByteDance

- 年份 / venue：2025 / SOSP 2025（`main`）
- 范围：`direct_checkpoint_and_recovery`；运行时 snapshot / state replication
- 页面 / SHA-256：18 pages / `6f664c3d3b6ffb93f0ab8a5385ed2ab3930d495639224af36921d1a2bacc816c`
- 链接：[venue](https://doi.org/10.1145/3731569.3764838)；[PDF source](https://i.cs.hku.hk/~cwu/papers/brwan-sosp25.pdf)；本地 `2025-sosp-byterobust.pdf`
- checkpoint 第一落点：本机与 peer machine 的 Host CPU memory。
- staging path：GPU model/optimizer shard → dual CPU buffers → D2H、serialization 与 peer send 三阶段重叠；peer shard 保存于 CPU memory。论文同时声明 Host-memory + local SSD 分层，但没有给出 CPU→SSD 的频率和完成协议。
- 最终介质：快速层为 local/peer Host DRAM；提及 local SSD tier；刻意避免远端 HDFS。
- redundancy：每个 shard 跨 PP/DP/TP failure group 备份到 peer；ZeRO-only 时退化为邻机备份，以抵抗 over-eviction。
- async boundary：D2H 使用独立 CUDA stream；backup 通信与 F/B、model/optimizer update 重叠。optimizer step 仍等待本 rank 自身 checkpoint save 完成。
- commit semantics：恢复选择 D2H 与 serialization 已完成的最新版本；本地 create 有等待边界，peer backup 可继续异步。论文没有定义 local SSD drain/commit。
- 核心机制：ByteRobust 将高频 Host checkpoint、跨并行组副本、warm standby、诊断和恢复统一到生产运行时。
- 对 AegisKV 的启示：这是最接近 AegisKV 算法 2/3 的参考：先按 failure domain 选副本，再由 α/β 控制 Host 与 SSD；图上必须把“本 rank safe”与“peer safe”拆开。
- 证据页：PDF pp. 5, 9–10, 13–14（Fig. 8–9、§6.3、实现）。
- 局限：SSD 层只被概述，缺少 drain 与 atomic publication；跨组策略假设 failure domain 可建模，且占用 2 TB 级 Host memory 与网络空隙。

### TC06 — PhoenixOS: Concurrent OS-level GPU Checkpoint and Restore with Validated Speculation

- 年份 / venue：2025 / SOSP 2025（`main`）
- 范围：`direct_system_checkpoint_restore`；运行时 snapshot / state replication
- 页面 / SHA-256：18 pages / `e20cb978be0e57fbcc5cb8136d8527840694f432cbabcb20b11c2b2a3e7f76ce`
- 链接：[venue](https://doi.org/10.1145/3731569.3764813)；[PDF source](https://ipads.se.sjtu.edu.cn/zh/publications/sosp25-wei.pdf)；本地 `2025-sosp-phoenixos.pdf`
- checkpoint 第一落点：可配置：CPU DRAM、local SSD，或另一台机器的 DRAM（RDMA）；论文评测多使用 Host DRAM。
- staging path：CRIU CPU state 与 GPU state 协调；GPU buffer 通过 soft CoW 或 recopy 并发复制到 backend，恢复采用 soft on-demand restore。
- 最终介质：CPU DRAM / local SSD / remote DRAM，取决于 backend。
- redundancy：没有内建 checkpoint replica/erasure code。
- async boundary：checkpoint data copy 与 GPU 执行并发；CoW 得到 t1 状态，recopy 在 t2 短暂停机补拷写集；异步调用仅在上一 checkpoint 未结束时才会阻塞。
- commit semantics：CoW 在并发 copy 完成后得到等价于 t1 的一致快照；recopy 在 quiesce+补拷后得到 t2 快照。论文保证执行一致性，但未定义 SSD fsync/跨组件持久 commit。
- 核心机制：PhoenixOS 以 validated speculation 构造 GPU soft-CoW/recopy/on-demand restore，实现透明 OS-level C/R。
- 对 AegisKV 的启示：AegisKV 可借鉴“安全状态时间点”而非复杂公式：算法 2 输出明确的 t1/t2 可恢复边界，算法 3 再调资源。
- 证据页：PDF pp. 4–7, 11–14, 18（Fig. 4、§4–§6、App. A.2）。
- 局限：间接寻址会使 speculation 失败并需保守处理；不同 backend durability 不同；未提供训练框架级分布式一致性。

### TC07 — ATTNChecker: Highly-Optimized Fault Tolerant Attention for Large Language Model Training

- 年份 / venue：2025 / PPoPP 2025（`main`）
- 范围：`recovery_avoidance`；恢复/诊断相关，但不定义新 checkpoint 写路径
- 页面 / SHA-256：15 pages / `709a6007bea605b49077dc43ae09c268d1944ef772b3d4668e22c151dee23e57`
- 链接：[venue](https://doi.org/10.1145/3710848.3710870)；[PDF source](https://arxiv.org/pdf/2410.11720)；本地 `2025-ppopp-attnchecker.pdf`
- checkpoint 第一落点：N/A：ATTNChecker 不创建模型 checkpoint。
- staging path：在 attention 运算内用 ABFT checksum/重算实时检测并修正 INF/NaN，目标是避免回滚到旧 checkpoint。
- 最终介质：N/A。
- redundancy：算法级 checksum 冗余，而非 checkpoint 数据副本。
- async boundary：在线检测与修正位于 attention critical path；没有 snapshot/persist 异步边界。
- commit semantics：N/A；修正通过校验通过而成立，不存在 checkpoint commit。
- 核心机制：针对 attention 的结构化 ABFT，减少因 INF/NaN 传播造成的训练回滚。
- 对 AegisKV 的启示：AegisKV 应把“重算/校验可覆盖的错误”从备份需求中扣除，作为算法 2 的故障类型过滤器。
- 证据页：PDF pp. 1–2, 5–10, 12。
- 局限：保护范围集中在 attention/特定数值错误；不能替代硬件、进程或整机故障的 checkpoint。

### TC08 — FT2: First-Token-Inspired Online Fault Tolerance on Output Tokens for Large Language Model Training

- 年份 / venue：2025 / HPDC 2025（`main`）
- 范围：`recovery_avoidance`；恢复/诊断相关，但不定义新 checkpoint 写路径
- 页面 / SHA-256：14 pages / `0db17ba06abbed04fe099f5663d8dfd8365d529c550ebde6cdb63b762b92cd6a`
- 链接：[venue](https://doi.org/10.1145/3731545.3731555)；[PDF source](https://lishanyang.github.io/HPDC25_FT2.pdf)；本地 `2025-hpdc-ft2.pdf`
- checkpoint 第一落点：N/A：FT2 是在线 token/关键层容错，不创建训练 checkpoint。
- staging path：利用 first-token 输入覆盖后续 token 的特性，对关键层做范围保护/纠错；无 GPU→Host→storage 路径。
- 最终介质：N/A。
- redundancy：关键层/首 token 复用带来的算法冗余，不是状态副本。
- async boundary：在线执行；没有 checkpoint 异步 drain。
- commit semantics：N/A。
- 核心机制：First-token-inspired critical-layer protection，用较低开销提高 LLM 输出 token 的故障容忍度。
- 对 AegisKV 的启示：算法 1 的 criticality 可结合 FT2 式算子保护概率，避免所有故障都落到同一种 backup 策略。
- 证据页：PDF pp. 1–2, 5–9, 12。
- 局限：面向输出/算子故障模型，不能恢复 optimizer、进程或节点状态。

### TC09 — Minder: Faulty Machine Detection for Large-scale Distributed Model Training

- 年份 / venue：2025 / NSDI 2025（`main`）
- 范围：`recovery_support_detection`；恢复/诊断相关，但不定义新 checkpoint 写路径
- 页面 / SHA-256：18 pages / `21eac11a7b2ca6dc16f1bbbbd1ee2aafb2b2def3e9c42dc952ddaa582f99096a`
- 链接：[venue](https://www.usenix.org/conference/nsdi25/presentation/deng)；[PDF source](https://www.usenix.org/system/files/nsdi25-deng.pdf)；本地 `2025-nsdi-minder.pdf`
- checkpoint 第一落点：N/A：Minder 只检测 faulty machine。
- staging path：采集监控 metric pattern → 自动检测异常机器 → 交给外部训练恢复系统。
- 最终介质：N/A。
- redundancy：无 checkpoint 冗余。
- async boundary：在线监控与检测；无 snapshot/persist。
- commit semantics：N/A。
- 核心机制：用故障前持续一段时间的可区分监控模式，在大规模训练停止前快速定位机器。
- 对 AegisKV 的启示：可作为算法 1/2 的 failure-probability 与 failure-domain 输入，而不是备份执行器。
- 证据页：PDF pp. 2, 5–10。
- 局限：定位后如何替换机器、选 checkpoint、恢复一致性均在论文范围外。

### TC10 — Holmes: Localizing Irregularities in LLM Training with Mega-scale GPU Clusters

- 年份 / venue：2025 / NSDI 2025（`main`）
- 范围：`recovery_support_diagnosis`；恢复/诊断相关，但不定义新 checkpoint 写路径
- 页面 / SHA-256：19 pages / `c5ffadfb6c2c569bb92438b7648ca69854ccf8873ef413f1f51e52c0155539ef`
- 链接：[venue](https://www.usenix.org/conference/nsdi25/presentation/yao)；[PDF source](https://www.usenix.org/system/files/nsdi25-yao.pdf)；本地 `2025-nsdi-holmes.pdf`
- checkpoint 第一落点：N/A：Holmes 诊断慢 iteration，不创建 checkpoint。
- staging path：通信 operator graph + 跨 iteration 分析 → 实时定位性能 irregularity。
- 最终介质：N/A。
- redundancy：无 checkpoint 冗余。
- async boundary：在线诊断；无 checkpoint pipeline。
- commit semantics：N/A。
- 核心机制：把通信算子依赖图用于 silent performance irregularity localization。
- 对 AegisKV 的启示：其 operator graph 可供算法 1 估计 slack，但不应混入算法 2 的 placement/commit 逻辑。
- 证据页：PDF pp. 2, 5–10。
- 局限：面向性能异常而非训练状态持久性；诊断成功不等于可恢复。

### TC11 — Evolution of Aegis: Fault Diagnosis for AI Model Training Service in Production

- 年份 / venue：2025 / NSDI 2025（`main`）
- 范围：`recovery_support_diagnosis`；恢复/诊断相关，但不定义新 checkpoint 写路径
- 页面 / SHA-256：18 pages / `7f4123e0ad3295bccf6a9e3ab7c73427d2b5e9bb615a95b0c3a8739b1460069a`
- 链接：[venue](https://www.usenix.org/conference/nsdi25/presentation/dong)；[PDF source](https://www.usenix.org/system/files/nsdi25-dong.pdf)；本地 `2025-nsdi-evolution-aegis.pdf`
- checkpoint 第一落点：N/A：Aegis（该论文）是训练云故障诊断系统。
- staging path：监控/collective instrumentation → failure localization、fail-slow 与交付前检查 → 外部 restart/recovery。
- 最终介质：N/A。
- redundancy：无 checkpoint 数据副本。
- async boundary：运行时诊断，不定义 checkpoint 异步边界。
- commit semantics：N/A。
- 核心机制：Aegis 从通用诊断演化到面向 AI collective 的生产故障定位。
- 对 AegisKV 的启示：可把诊断置信度映射为 AegisKV 的风险输入；注意与本项目 AegisKV 名称相同但机制不同。
- 证据页：PDF pp. 2, 5–13。
- 局限：减少 diagnosis idle time，但状态保存、备份层级与恢复提交由其他系统负责。

### TC12 — Training with Confidence: Catching Silent Errors in Deep Learning Training with Automated Proactive Checks

- 年份 / venue：2025 / OSDI 2025（`main`）
- 范围：`recovery_support_sdc_detection`；恢复/诊断相关，但不定义新 checkpoint 写路径
- 页面 / SHA-256：18 pages / `cb8e8c16bdf93ba0f6c6bfdf7715e6ead12824c59c903b8cccd2735fa9f27b8b`
- 链接：[venue](https://www.usenix.org/conference/osdi25/presentation/jiang)；[PDF source](https://www.usenix.org/system/files/osdi25-jiang.pdf)；本地 `2025-osdi-traincheck.pdf`
- checkpoint 第一落点：N/A：TrainCheck 检查训练 invariant，不生成 checkpoint。
- staging path：自动推断 invariant → 每 iteration 主动检查 silent training error → 提供 debug context。
- 最终介质：N/A。
- redundancy：无 checkpoint 冗余。
- async boundary：在线检查；无持久化 pipeline。
- commit semantics：N/A。
- 核心机制：以训练特定 invariant 及 proactive checks 捕获静默软件/配置错误。
- 对 AegisKV 的启示：算法 2 可把“checkpoint 是否可信”作为 admission 条件，避免复制已经污染的状态。
- 证据页：PDF pp. 2, 5, 10–12。
- 局限：不能处理所有硬件/系统故障；检测后的回滚点和状态保护不在范围内。

### TC13 — Mycroft: Tracing Dependencies in Collective Communication Towards Reliable LLM Training

- 年份 / venue：2025 / SOSP 2025（`main`）
- 范围：`recovery_support_diagnosis`；恢复/诊断相关，但不定义新 checkpoint 写路径
- 页面 / SHA-256：16 pages / `0577ac16ae615dd0886742fcab856d47d22ffb90f9eff19c3ef4f3e04c3def5c`
- 链接：[venue](https://doi.org/10.1145/3731569.3764848)；[PDF source](https://minlanyu.seas.harvard.edu/writeup/sosp25.pdf)；本地 `2025-sosp-mycroft.pdf`
- checkpoint 第一落点：N/A：Mycroft 做 collective tracing/根因定位。
- staging path：轻量 instrumentation → flow/data-chunk dependency trace → trigger 后在线 root-cause analysis。
- 最终介质：N/A。
- redundancy：无 checkpoint 副本。
- async boundary：追踪按 trigger 开启；无 snapshot/persist。
- commit semantics：N/A。
- 核心机制：暴露 collective 内部 flow 与 data-chunk dependency，定位 hang/性能异常。
- 对 AegisKV 的启示：其 failure-domain/dependency 可帮助算法 2 避免把主本与备份放在同一通信 blast radius。
- 证据页：PDF pp. 1–2, 5–10, 13。
- 局限：解决 observability 而非 state recovery；依赖通信库 instrumentation。

### TC14 — TrainVerify: Equivalence-Based Verification for Distributed LLM Training

- 年份 / venue：2025 / SOSP 2025（`main`）
- 范围：`recovery_support_prevention`；恢复/诊断相关，但不定义新 checkpoint 写路径
- 页面 / SHA-256：21 pages / `61b4791fc5568aa16705615627fadadc0eeacd88347be79962b036a8d77b641d`
- 链接：[venue](https://doi.org/10.1145/3731569.3764850)；[PDF source](https://arxiv.org/pdf/2506.15961)；本地 `2025-sosp-trainverify.pdf`
- checkpoint 第一落点：N/A：TrainVerify 验证 execution plan 等价性。
- staging path：logical model/plan → symbolic DFG → shape reduction + staged SMT verification。
- 最终介质：N/A。
- redundancy：无 checkpoint 冗余。
- async boundary：部署/编译期验证；无运行时 checkpoint pipeline。
- commit semantics：N/A。
- 核心机制：证明分布式并行 execution plan 与逻辑模型等价，避免错误计划污染训练。
- 对 AegisKV 的启示：可以在算法 2 执行 layout change 前做 correctness gate，但不能充当 durability 证明。
- 证据页：PDF pp. 1–2, 5–10。
- 局限：不检测运行时硬件故障，也不保存/恢复训练状态。

### TC15 — LLMTailor: A Layer-wise Tailoring Tool for Efficient Checkpointing of Large Language Models

- 年份 / venue：2025 / PDSW at SC 2025（`peer-reviewed workshop`）
- 范围：`direct_checkpoint`；格式、后处理、读取或经验 I/O
- 页面 / SHA-256：9 pages / `0ca90bc1a2b61c9442ca4877cc7e49b57f934a167aa0e2866d8bb3b03b477389`
- 链接：[venue](https://doi.org/10.1145/3731599.3767515)；[PDF source](https://arxiv.org/pdf/2602.22158)；本地 `2025-pdsw-llmtailor.pdf`
- checkpoint 第一落点：N/A（在线 snapshot）：LLMTailor 读取已有本地 checkpoint files，并输出新的本地 composite checkpoint。
- staging path：多个历史 model/optimizer checkpoint files → CPU multiprocessing 按 layer/parameter group 拆选 → 顺序写出新 model、optimizer shards 与 config。
- 最终介质：本地文件系统中的 Frankenstein checkpoint。
- redundancy：跨时间版本保留不同 layer，可用于组合；没有副本/纠删码。
- async boundary：离线 merge 工具，不与训练 D2H 或 storage drain 重叠。
- commit semantics：所有 weight、optimizer shard、辅助 layer 和 config 都写完后才可恢复；工具保持 load/write 顺序，但未提供 crash-atomic publication。
- 核心机制：用 layer-wise partial checkpoint + merge 探索减少保存体积，支持可继续训练的完整 optimizer state。
- 对 AegisKV 的启示：对 AegisKV 的价值在于“备份粒度可按 layer 选择”，但算法 2 必须增加质量/陈旧性 gate，不能只按 size 最小化。
- 证据页：PDF pp. 1–7, 9（§4–§5、结论、artifact）。
- 局限：仅支持本地 checkpoint；规则化 selective saving 可能影响质量；不是生产在线 checkpoint engine。

### TC16 — LLM training in practice: insights from 85,000 checkpoints

- 年份 / venue：2025 / PDSW at SC 2025（`work-in-progress`）
- 范围：`direct_empirical_wip`；格式、后处理、读取或经验 I/O
- 页面 / SHA-256：2 pages / `942d98dd76a0e637d8cc069946d0a2e1ea44a200b1510646bd6c3be04f725ecf`
- 链接：[venue](https://sc25.supercomputing.org/proceedings/workshops/workshop_pages/ws_pdswwip108.html)；[PDF source](https://pdsw.org/pdsw25/papers/wips/ws_pdswwip108s2-file1.pdf)；本地 `2025-pdsw-85000-checkpoints.pdf`
- checkpoint 第一落点：聚合观测显示现代作业先写 node-local storage；论文未说明该介质是 DRAM 还是 SSD。
- staging path：训练状态 → node-local storage → 训练恢复后异步 drain 到全局 shared storage。
- 最终介质：共享存储平台（基于 VAST telemetry 的观测）。
- redundancy：论文不披露各作业副本策略。
- async boundary：global storage 看到的是训练已恢复后的 drain；用 checkpoint-overlap（drain 时间/间隔）而不是 GPU 峰值来规划带宽。
- commit semantics：观测数据没有记录 per-job create/commit 元数据；node-local 完成与 shared-storage durability 不能等同。
- 核心机制：从 40 个生产作业、85,000+ checkpoint 反推存储需求，指出共享层带宽通常只需支持低 overlap 的平滑 drain。
- 对 AegisKV 的启示：这直接支持 AegisKV 算法 3 用窗口化带宽利用率调 β，而不是按 GPU 总数静态预留峰值带宽。
- 证据页：PDF pp. 1–2（Methods 与 Preliminary findings）。
- 局限：2 页 WIP；供应商 telemetry、介质与软件栈不透明；是经验趋势而非可复现 checkpoint 协议。

### TC17 — Checkmate: Zero Performance Overhead Model Checkpointing via Network Gradient Replication

- 年份 / venue：2026 / NSDI 2026（`main`）
- 范围：`direct_checkpoint`；运行时 snapshot / state replication
- 页面 / SHA-256：21 pages / `2eb4d563a0104445a5024dabca7cb2388b77e15742255ebc94c11a6f56ac33de`
- 链接：[venue](https://www.usenix.org/conference/nsdi26/presentation/bhardwaj)；[PDF source](https://www.usenix.org/system/files/nsdi26-bhardwaj.pdf)；本地 `2026-nsdi-checkmate.pdf`
- checkpoint 第一落点：独立 CPU shadow cluster 的 Host DRAM；没有 GPU snapshot D2H。
- staging path：AllReduce reduced gradients → programmable switch tag/multicast → shadow NIC/CPU bucket → CPU optimizer 更新 Host-memory model/optimizer replica。
- 最终介质：原型 checkpoint 位于 shadow Host DRAM；评测用 in-memory nullfs 排除持久存储，未实现磁盘提交。
- redundancy：模型按 DP shard 分布在 shadow nodes；PFC/lossless delivery 保证梯度到达，但没有 shadow-state 的持久副本。
- async boundary：GPU 完全不参与 checkpoint；shadow optimizer 必须在下一 GPU iteration 开始前完成，属于旁路但有 pace deadline。
- commit semantics：收齐本 iteration 所有 tagged gradient buckets 并完成 shadow optimizer step 后，该 iteration 的内存 checkpoint 才可恢复；无 disk commit/shadow-cluster failure protocol。
- 核心机制：Checkmate 用网络 multicast 复制梯度，在 CPU shadow cluster 每 iteration 重建 checkpoint。
- 对 AegisKV 的启示：可把算法 2 的 placement 扩展为“旁路重建到 Host”，并把网络 port/CPU optimizer deadline 纳入资源约束。
- 证据页：PDF pp. 2–8, 10–14（Fig. 3–5、§4.2–§4.4、讨论）。
- 局限：依赖 lossless/PFC 与 switch programmability；FSDP/ZeRO 仍是未来工作；没有持久层与 shadow failure 容忍。

### TC18 — Sparse Checkpointing for Fast and Reliable MoE Training

- 年份 / venue：2026 / NSDI 2026（`main`）
- 范围：`direct_checkpoint`；运行时 snapshot / state replication
- 页面 / SHA-256：22 pages / `eb25c8fda8daa0bb2e2927ca3890b7414a20ee7a4674dc383d47bfde2e0b8fd9`
- 链接：[venue](https://www.usenix.org/conference/nsdi26/presentation/gandhi)；[PDF source](https://www.usenix.org/system/files/nsdi26-gandhi.pdf)；本地 `2026-nsdi-moevement.pdf`
- checkpoint 第一落点：先到本机 CPU memory，再到 r 个 peer nodes 的 remote CPU memory。
- staging path：每 iteration 的部分 expert/non-expert 状态 GPU→local CPU；异步复制到 r 个 peer Host；pipeline 边界 activation/gradient logs 留在邻居 Host memory。
- 最终介质：distributed in-memory checkpoint（peer Host DRAM），无 disk/remote object store。
- redundancy：每个 sparse snapshot 复制到 r 个 peer；系统保留一个已提交 sparse checkpoint 和一个 in-flight 版本。
- async boundary：每 iteration 小 snapshot 与训练完全重叠；Wsparse 个 iteration 组成一个窗口。log GC 仅在新窗口 checkpoint 已持久复制后发生。
- commit semantics：当窗口内全部 sparse snapshots 都在 r 个 peers 上“durably replicated”时，整个 sparse checkpoint 才算 persisted；随后才回收旧版本。
- 核心机制：MoEvement 以 sparse operator windows、sparse-to-dense replay 和 upstream logging 同时降低保存与局部恢复成本。
- 对 AegisKV 的启示：算法 2 可直接采用“窗口 commit”：K 内各请求/层逐步备份，K+1 只有在窗口副本齐全后才能回收；算法 3 再根据压力调 α/β。
- 证据页：PDF pp. 2–8, 12–13, 20–22（Fig. 5–9、§3.2–§3.6）。
- 局限：只在 Host memory 提供 durability，peer/common-mode failure 风险仍在；恢复需重放且当前主要针对 MoE；Host 内存增量明显。

### TC19 — AdaCheck: An Adaptive Checkpointing System for Efficient LLM Training with Redundancy Utilization

- 年份 / venue：2026 / FAST 2026（`main`）
- 范围：`direct_checkpoint`；运行时 snapshot / state replication
- 页面 / SHA-256：20 pages / `513edfc55023b3563194d18e0a10b8c95fdffb2d6f427c226538504c5941bcdb`
- 链接：[venue](https://www.usenix.org/conference/fast26/presentation/liu-weijie)；[PDF source](https://www.usenix.org/system/files/fast26-liu-weijie.pdf)；本地 `2026-fast-adacheck.pdf`
- checkpoint 第一落点：checkpoint group 内其他 worker 的 remote CPU memory。
- staging path：无/低冗余状态的 parameter 或 gradient 经训练网络发给 peer CPU；remote CPU optimizer 每 iteration 应用 gradient，维护可直接恢复的完整状态；另有 background full checkpoint 写 remote persistent storage。
- 最终介质：快速层为 peer Host DRAM；灾难恢复层为 remote persistent storage（具体介质未披露）。
- redundancy：按用户 failure-tolerance factor k 分 checkpoint groups，组内互换状态；现存训练 replica 也被计入 redundancy，full persistent checkpoint 覆盖 catastrophic failure。
- async boundary：incremental checkpoint communication 与模型计算重叠；完整 checkpoint 由专用后台线程 non-blocking 保存到持久层。
- commit semantics：快速版本在 peer 收到并由 remote CPU optimizer 应用本 iteration 更新后可恢复；完整持久 checkpoint 的跨 rank atomic publication/完成检测未定义。
- 核心机制：AdaCheck 先在线检测 tensor redundancy，再以 gradient-based incremental remote updating 只保存必要状态，并辅以后台 full checkpoint。
- 对 AegisKV 的启示：这是 AegisKV 资源管理建模的强参考：把 replica 数 k、Host 容量、网络带宽和 persistent full-checkpoint 周期作为明确资源，而不是堆叠复杂目标函数。
- 证据页：PDF pp. 4–10, 12–13（Fig. 9–10、§7–§9.5）。
- 局限：恢复概率依赖 k 与独立故障假设；remote persistent save 的格式、commit 与频率未细化；需要 CPU optimizer 与并行策略适配。

### TC20 — GPU Checkpoint/Restore Made Fast and Lightweight

- 年份 / venue：2026 / FAST 2026（`main`）
- 范围：`direct_system_checkpoint_restore`；运行时 snapshot / state replication
- 页面 / SHA-256：17 pages / `464a16a86dd4cd336d69f3171625b2cfbf1d35b01a2630a80806cea16ea7b4c1`
- 链接：[venue](https://www.usenix.org/conference/fast26/presentation/zeng)；[PDF source](https://www.usenix.org/system/files/fast26-zeng.pdf)；本地 `2026-fast-gcr.pdf`
- checkpoint 第一落点：当前实现为 CPU memory。
- staging path：checkpoint 前 cudaDeviceSynchronize；GPU data buffers 用 cudaMemcpyAsync 拷到 CPU，GPU control state 由 driver-integrated C/R 保存；增量版本与前一 full checkpoint 在 CPU memory 合并。
- 最终介质：CPU DRAM；SSD/remote memory 仅列为 future backend。
- redundancy：无数据副本/纠删码。
- async boundary：dirty-template 的 CPU shadow execution 与 GPU kernel 并行；真正 checkpoint 前仍同步 kernel 并暂停执行，故 memcpyAsync 不等于训练异步 checkpoint。
- commit semantics：data buffers 与 control state 都保存、地址映射一致且增量已合并后形成可恢复 checkpoint；没有持久 commit。
- 核心机制：GCR 以 control/data separation、driver+interception 混合 C/R 和 CPU shadow dirty tracking 实现快速透明 GPU C/R。
- 对 AegisKV 的启示：图中要严格区分“异步识别 dirty state”与“同步 create checkpoint”；算法 3 不应把 async API 调用误判为零阻塞。
- 证据页：PDF pp. 4–10, 12–14（§4–§6、实现与讨论）。
- 局限：当前 checkpoint 只在 CPU memory；训练 checkpoint 仍会 stall；增量优化主要围绕 inference KV cache，SSD/remote backend 未实现。

### TC21 — TrainMover: An Interruption-Resilient Runtime for ML Training

- 年份 / venue：2026 / OSDI 2026（`main`）
- 范围：`recovery_and_state_migration`；恢复/诊断相关，但不定义新 checkpoint 写路径
- 页面 / SHA-256：19 pages / `c7dfb3c467f5206ce231c11fbdd86b5ea37cbc020007db40ac1e98aa95fcd1ff`
- 链接：[venue](https://www.usenix.org/conference/osdi26/presentation/lao)；[PDF source](https://www.usenix.org/system/files/osdi26-lao.pdf)；本地 `2026-osdi-trainmover.pdf`
- checkpoint 第一落点：N/A：TrainMover 不提供新的 checkpoint engine；joiner 可从 leaver/DP replica/remote CPU checkpoint/remote storage 取状态。
- staging path：expected event：leaver→joiner 一对一 RDMA state transfer；unexpected event：优先从 stayer redundancy/remote CPU checkpoint 取状态，否则从 remote storage checkpoint 恢复。
- 最终介质：沿用外部 checkpoint 的 Host 或远端存储介质。
- redundancy：spare GPU、DP redundancy 和 general standby；并非新 checkpoint replica。
- async boundary：sandbox warmup 与两阶段 CCL setup 在后台重叠；最终 membership switchover 与 state readiness 是短暂停机边界。
- commit semantics：joiner 状态、通信拓扑和 CCL 都 ready 后才切换角色；外部 checkpoint 的 create/commit 不在论文范围。
- 核心机制：TrainMover 把初始化、CCL 建链和角色 warmup 移出 interruption critical path，实现低停机迁移。
- 对 AegisKV 的启示：算法 2 应把“直接迁移/利用现有 replica”作为比新增备份更便宜的动作；算法 3 管理 standby 与网络 headroom。
- 证据页：PDF pp. 2, 4–10, 15–16。
- 局限：依赖现有状态来源；异步训练不支持；应用正确性与 checkpoint durability 均交给用户/外部系统。

### TC22 — RobustRL: Role-Based Fault Tolerance System for RL Post-Training

- 年份 / venue：2026 / OSDI 2026（`main`）
- 范围：`recovery_and_role_state`；运行时 snapshot / state replication
- 页面 / SHA-256：19 pages / `c0c9bb213775bc5742cca3e209a98ebeec915d5e3405b27ce51556994f314ff7`
- 链接：[venue](https://www.usenix.org/conference/osdi26/presentation/chen-zhenqian)；[PDF source](https://www.usenix.org/system/files/osdi26-chen-zhenqian.pdf)；本地 `2026-osdi-robustrl.pdf`
- checkpoint 第一落点：ByteCheckpoint 的 pinned local Host memory。
- staging path：每个 RL step：GPU shard → pinned Host memory（短阻塞）→ 异步写 HDFS；检测异常时可预取 HDFS shard 到本地 memory，再恢复 trainer。
- 最终介质：HDFS。
- redundancy：rollout replicas/warm standby 提供角色可用性；checkpoint 数据副本由 HDFS 后端负责，论文未披露因子。
- async boundary：只阻塞 trainer 的 GPU→Host；memory→HDFS 非阻塞且与 rollout 继续生成重叠。
- commit semantics：Host snapshot 完成不等于 durable；只有 HDFS 写完才可跨机恢复。RobustRL 沿用 ByteCheckpoint，但未单独描述 distributed atomic publication。
- 核心机制：RobustRL 以 role-based Detect-Restart-Reconnect 隔离 trainer/rollout failure，并用 per-step checkpoint 保持权重一致。
- 对 AegisKV 的启示：对 AegisKV 的启示是请求/角色 slack 可决定 Host→disk 是否来得及；长 RL step 允许更积极的 β drain。
- 证据页：PDF pp. 5–7, 10, 12–14（§5.1、Fig. 20–21）。
- 局限：checkpoint 机制来自 ByteCheckpoint 而非论文新贡献；per-step 合理性依赖 RL step 足够长；HDFS failure/commit 细节未评测。

### TC23 — SDCs in the Wild: Characterizing and Diagnosing SDC-Defective GPUs in Production LLM Training

- 年份 / venue：2026 / OSDI 2026（`main`）
- 范围：`recovery_support_sdc_diagnosis`；恢复/诊断相关，但不定义新 checkpoint 写路径
- 页面 / SHA-256：20 pages / `9d5ee870c3574fd83833eb41acd5e1e9bc3179f32bb2e853bc7616c8c8b3948b`
- 链接：[venue](https://www.usenix.org/conference/osdi26/presentation/zheng)；[PDF source](https://www.usenix.org/system/files/osdi26-zheng.pdf)；本地 `2026-osdi-sdcs-in-the-wild.pdf`
- checkpoint 第一落点：N/A：SDCHUNTER 消费“最新已验证 checkpoint”做 deterministic replay，不创建 checkpoint。
- staging path：verified checkpoint + 原始 input/communication trace → 两阶段 deterministic replay → 隔离 SDC GPU → 训练从 verified checkpoint 恢复。
- 最终介质：沿用外部 checkpoint 存储。
- redundancy：DP replay replicas 用于诊断投票，不是 checkpoint replica。
- async boundary：Phase 1 阻塞在线恢复；Phase 2/硬件确认可离线继续。无 checkpoint drain。
- commit semantics：只有 verified checkpoint 才被允许作为恢复源；create/commit 由外部系统负责。
- 核心机制：生产 workload 精确 replay 与分层 signature 比较，识别 data-dependent GPU SDC。
- 对 AegisKV 的启示：AegisKV 应给备份附加 validity 状态：未通过 SDC/一致性 gate 的副本即使持久化也不能计入保护度。
- 证据页：PDF pp. 2–3, 5, 8, 10–12。
- 局限：需要 deterministic training 与可信 reference replica；无法解决所有 replica 同样损坏；checkpoint 生成不在范围。

### TC24 — Safeguarding LLM Training at Scale: Online SDC Detection and Insights from 35 Million GPU Hours

- 年份 / venue：2026 / OSDI 2026（`main`）
- 范围：`recovery_support_sdc_detection`；恢复/诊断相关，但不定义新 checkpoint 写路径
- 页面 / SHA-256：17 pages / `17e76c1e6343e660ac955c023bb30c92d8d4297b10616a7fe2430bec3ecfd3dc`
- 链接：[venue](https://www.usenix.org/conference/osdi26/presentation/lei)；[PDF source](https://www.usenix.org/system/files/osdi26-lei.pdf)；本地 `2026-osdi-aegis-sdc.pdf`
- checkpoint 第一落点：N/A：AEGIS 是在线 SDC detection framework。
- staging path：轻量 cSensor 感知可疑 corruption → cVerifier 确认 → 隔离 faulty GPU/触发外部恢复。
- 最终介质：N/A。
- redundancy：检测交叉验证，不是 checkpoint 副本。
- async boundary：在线检测；无 snapshot/persist。
- commit semantics：N/A。
- 核心机制：以 cSensor-cVerifier 两阶段抽象在 0.86% overhead 下发现生产 SDC。
- 对 AegisKV 的启示：可把 verifier 信号作为算法 2 的“副本是否可信”与算法 3 的风险突增事件。
- 证据页：PDF pp. 2, 5–10, 14。
- 局限：只解决 SDC detection；恢复点、备份位置与持久一致性不在范围。

### TC25 — OpGuard: Bitwise Alignment for Precise and General Debugging of Production LLM Training

- 年份 / venue：2026 / OSDI 2026（`main`）
- 范围：`recovery_support_debugging`；恢复/诊断相关，但不定义新 checkpoint 写路径
- 页面 / SHA-256：23 pages / `28413e346fdc24d5397f5002cffdc3a6f31d791c10a0e6836dc2d6217b85b778`
- 链接：[venue](https://www.usenix.org/conference/osdi26/presentation/zhou-ziming)；[PDF source](https://www.usenix.org/system/files/osdi26-zhou-ziming.pdf)；本地 `2026-osdi-opguard.pdf`
- checkpoint 第一落点：N/A：OpGuard 对两次训练执行做 bitwise alignment/debugging。
- staging path：operator-boundary fingerprints → schedule-tolerant mapping → 找到首个 bitwise mismatch。
- 最终介质：N/A。
- redundancy：对照执行提供 correctness oracle，不是 checkpoint redundancy。
- async boundary：在线/重放调试；无 checkpoint pipeline。
- commit semantics：N/A。
- 核心机制：以 semantic-stable operator boundary 和 first mismatch 作为生产 LLM debugging primitive。
- 对 AegisKV 的启示：算法 2 可在备份前记录轻量 fingerprint，使恢复后可快速验证副本，而无需复制更多公式。
- 证据页：PDF pp. 2, 4–7, 11–12。
- 局限：依赖可控制的 benign nondeterminism 和可比较执行；不提供状态恢复。

### TC26 — Sparing Strategies to Minimize Reliability Impact On Large Training Jobs

- 年份 / venue：2026 / MLSys 2026（`main`）
- 范围：`recovery_capacity_planning`；恢复/诊断相关，但不定义新 checkpoint 写路径
- 页面 / SHA-256：14 pages / `e8e5216092f6fe9c8168948a023452164284c23cba601534a9fa0b0d550e6793`
- 链接：[venue](https://proceedings.mlsys.org/paper_files/paper/2026/hash/437bc4ccafd3fc6d4289bd10940be42b-Abstract-Conference.html)；[PDF source](https://proceedings.mlsys.org/paper_files/paper/2026/file/437bc4ccafd3fc6d4289bd10940be42b-Paper-Conference.pdf)；本地 `2026-mlsys-sparing-strategies.pdf`
- checkpoint 第一落点：N/A：论文把 checkpoint save/restart 当成模型参数，不实现存储路径。
- staging path：抽象为 checkpoint period、save time、detect time、restart time，与 tray/block spare、MTBF/MTTR、placement constraint 一起进入 Goodput/CETT 模型。
- 最终介质：N/A。
- redundancy：建模 intra-block spare tray 与 inter-block spare；checkpoint redundancy 由外部方案决定。
- async boundary：仅分析同步 recovery；未给具体 async pipeline。
- commit semantics：N/A。
- 核心机制：用闭式近似、Markov chain 与模拟器做 GPU sparing/repair/placement 的资源管理决策。
- 对 AegisKV 的启示：这是把 AegisKV 专利建模为经典资源管理问题的直接参考：最大化 Goodput/CETT，决策变量用 Host 容量、带宽、spare、failure domain，复杂理论退到离线配置器。
- 证据页：PDF pp. 2–7, 10, 13–14。
- 局限：示例参数不代表真实生产值；checkpoint 路径、数据一致性和异步提交均被抽象掉。

### TC27 — EROICA: Online Performance Troubleshooting for Large-scale Model Training

- 年份 / venue：2026 / NSDI 2026（`main`）
- 范围：`recovery_support_diagnosis`；恢复/诊断相关，但不定义新 checkpoint 写路径
- 页面 / SHA-256：19 pages / `ff87208d83a94f61ad3ac2711f310f19c9995dee7b9ca164a3f82359046c6e45`
- 链接：[venue](https://www.usenix.org/conference/nsdi26/presentation/guan-yu)；[PDF source](https://www.usenix.org/system/files/nsdi26-guan-yu.pdf)；本地 `2026-nsdi-eroica.pdf`
- checkpoint 第一落点：N/A：EROICA 是在线性能 troubleshooting system。
- staging path：全机在线 profiling + differential observability → 定位软硬件性能根因。
- 最终介质：N/A。
- redundancy：无 checkpoint replica。
- async boundary：在线 profiling；无 checkpoint pipeline。
- commit semantics：N/A。
- 核心机制：在约 100,000 GPU 生产集群做细粒度、全覆盖性能诊断。
- 对 AegisKV 的启示：其在线 profile 可给算法 3 提供 Host/网络/存储实际服务率，替代纯理论带宽估计。
- 证据页：PDF pp. 2–8。
- 局限：诊断性能问题，不创建或恢复训练 state。

### TC28 — Teaching the Old Dog New Tricks: Building Efficient Data Pipelines for Large-Scale LLM Pre-Training

- 年份 / venue：2026 / OSDI 2026（`main`）
- 范围：`checkpoint_loading_and_replication`；格式、后处理、读取或经验 I/O
- 页面 / SHA-256：18 pages / `61b6cb4de630b8c8d62b9241200a5ae51cb418c65e3ea5f9f34e83c0dd1af3d6`
- 链接：[venue](https://www.usenix.org/conference/osdi26/presentation/chen-luofan)；[PDF source](https://www.usenix.org/system/files/osdi26-chen-luofan.pdf)；本地 `2026-osdi-checkpoint-loading-pipeline.pdf`
- checkpoint 第一落点：N/A（save path）：输入 checkpoint 已在 persistent storage。
- staging path：remote/cross-DC checkpoint namespace → 预测复制到目标 DC；启动前把 hot metadata/replicated-tensor files 从默认 3 replicas 扩到约 128 ephemeral replicas → ranks 并发加载。
- 最终介质：分布式持久存储节点；TTL 到期自动回收临时副本。
- redundancy：默认 3 副本；按 world size/access fan-out 对热点文件 proactive replica expansion。
- async boundary：复制在 recovery/startup 前完成；真正 checkpoint loading 是所有 ranks 等待最慢者的同步 barrier。
- commit semantics：replica readiness 在 recovery 开始前成立；论文不定义 checkpoint save/commit，仅定义 load 前的副本准备。
- 核心机制：预测跨 DC checkpoint replication、热点文件预复制与 storage-side transform offload。
- 对 AegisKV 的启示：算法 2 不应只考虑写入：恢复 fan-out 会反向决定 β 和 storage replica；算法 3 可在 K+1 前预扩热点。
- 证据页：PDF pp. 2–4, 8–10, 13–14。
- 局限：只优化读取/数据管线；依赖可预测 access pattern 与 storage namespace；不会改善初始 checkpoint commit。

### TC29 — Di-PS: System-Algorithm Co-Design for Asynchronous and Heterogeneous Cross-cluster LLM Training at Scale

- 年份 / venue：2026 / NSDI 2026（`main`）
- 范围：`recovery_and_state_replication`；运行时 snapshot / state replication
- 页面 / SHA-256：20 pages / `537de93c8b94231a0223bc8580972ce913381b55c87936110759e40c7570352a`
- 链接：[venue](https://www.usenix.org/conference/nsdi26/presentation/li-shengwei)；[PDF source](https://www.usenix.org/system/files/nsdi26-li-shengwei.pdf)；本地 `2026-nsdi-di-ps.pdf`
- checkpoint 第一落点：模型状态本来就位于 follower PS 的 CPU DRAM；论文未披露异步 checkpoint 的第一持久落点。
- staging path：每个 outer optimizer step 后，distributed follower PS 从其 CPU-resident shard 异步保存 snapshot；leader 只保存 <100 KB metadata；仅保留最近几个版本。
- 最终介质：未说明（只写 checkpoint/snapshot storage）。
- redundancy：follower PS 按 layer 分片而非复制；leader-follower 控制面可重启，但 checkpoint 数据副本因子未披露。
- async boundary：follower checkpoint 与 control/data flow 重叠；leader failure 时 inner training non-blocking 继续、跳过 outer update。
- commit semantics：leader 用 latest checkpoint 重启 failed follower；论文未给 storage durability、跨 followers atomic point 或 publish protocol。
- 核心机制：Di-PS 以 CPU parameter-server 全局状态、异步 outer optimizer 与 follower snapshot 实现跨集群韧性。
- 对 AegisKV 的启示：提醒 AegisKV：CPU-native control state 与 GPU state 不能套同一 D2H 模型；算法 2 需要按 state owner 区分路径。
- 证据页：PDF pp. 6–9（Fig. 4、§4.1、§4.3）。
- 局限：destination/commit 语义不透明；distributed followers 的一致 cut 未描述；它是 cross-cluster optimizer 系统而非通用 training checkpoint engine。

### TC30 — FLARE: Anomaly Diagnostics for Divergent LLM Training in GPU Clusters of Thousand-Plus Scale

- 年份 / venue：2026 / NSDI 2026（`main`）
- 范围：`recovery_support_diagnosis`；恢复/诊断相关，但不定义新 checkpoint 写路径
- 页面 / SHA-256：16 pages / `f8ea86442448a29b0a48a7ee232b9dd28865fcf6fe3ab4bf1c4cfd5ddfa86e24`
- 链接：[venue](https://www.usenix.org/conference/nsdi26/presentation/cui)；[PDF source](https://www.usenix.org/system/files/nsdi26-cui.pdf)；本地 `2026-nsdi-flare.pdf`
- checkpoint 第一落点：N/A：FLARE 是全栈 anomaly diagnostics。
- staging path：轻量 tracing daemon → backend-extensible trace → 自动 diagnostic engine 定位 performance regression/divergence。
- 最终介质：N/A。
- redundancy：无 checkpoint redundancy。
- async boundary：在线 tracing；无 snapshot/persist。
- commit semantics：N/A。
- 核心机制：覆盖训练全栈的 anomaly diagnostics，已在 6,000 GPUs 部署。
- 对 AegisKV 的启示：可把诊断输出变成算法 3 的事件输入，让 α/β 对故障与资源退化做窗口化调整。
- 证据页：PDF pp. 2–8, 11。
- 局限：诊断覆盖不等于故障恢复；checkpoint create/load 都不在范围。

## 文件级审计说明

- `download_validation.csv` 保存原始下载 URL、来源类型、bytes、pages、SHA-256 和 PDF magic。
- `manifest.csv` 在其基础上加入逐篇语义字段。所有 `N/A` 都表示论文未定义该机制，而不是审读遗漏。
- PDF 页码均为文件中的 1-based PDF page number，便于直接跳转核对；它可能与论文印刷页码不同。
- 2026 论文均来自会议官方 proceedings 页面或其官方 PDF；2025 的作者 accepted/VOR PDF 在 `pdf_source` 中单独标明。
