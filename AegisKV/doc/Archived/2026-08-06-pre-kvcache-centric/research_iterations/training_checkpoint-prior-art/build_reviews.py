#!/usr/bin/env python3
"""Build the audited manifest and per-paper semantic review.

The download-level facts come from download_validation.csv.  The semantic
fields below are deliberately curated from the PDFs, rather than inferred from
titles or abstracts.  "N/A" is used when a paper consumes checkpoints or
improves recovery but does not define a checkpoint creation path.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def review(
    system_group: str,
    first_destination: str,
    staging_path: str,
    final_medium: str,
    redundancy: str,
    async_boundary: str,
    commit_semantics: str,
    key_mechanism: str,
    evidence_pages: str,
    limitations: str,
    aegiskv_insight: str,
    host_to_storage_class: str,
) -> dict[str, str]:
    return locals()


REVIEWS = {
    "TC01": review(
        "runtime_snapshot_or_replication",
        "本机 pinned Host DRAM（CPU buffer pool）。",
        "GPU 状态按 chunk 经 DMA 拷到 pinned DRAM；DRAM chunk 随即由多线程流水写入 SSD 或 PMEM，可同时保留多个 in-flight checkpoint。",
        "本机/云持久 SSD，或 PMEM。",
        "没有跨机副本；但地址队列保证最近 N+1 个版本中至少一个已持久化版本不被覆盖。",
        "训练与 D2H、DRAM→storage 重叠；多个 checkpoint 可并发，snapshot 与 persist 还可按 chunk 流水。",
        "所有 writer 完成后，PMEM 路径逐线程 fence，SSD 路径由主线程 msync；随后以计数器/CAS 更新 latest-checkpoint pointer。创建异步，但可恢复版本要到该持久化与指针发布边界后才成立。",
        "PCcheck 用并发 checkpoint 数、chunk 数、DRAM buffer 和 writer 数共同调度 PCIe、DRAM 与存储带宽。",
        "PDF pp. 5–8, 12–13（Fig. 6–7、§3.3、§4.1、结论）。",
        "持久层仍是单机故障域；DRAM 容量、PCIe 与 SSD 带宽会限制并发度；PMEM 设备可获得性有限。",
        "可把 AegisKV 的 α 解释为 Host 快速层预算、β 解释为持久层 drain 预算；提交状态必须像 PCcheck 一样与“已 D2H”分离。",
        "host_to_persistent_explicit",
    ),
    "TC02": review(
        "runtime_snapshot_or_replication",
        "独立 CPU checkpoint server 的 Host DRAM，而非训练节点 Host memory。",
        "交换机镜像 AllReduce 完整梯度 → checkpoint server NIC/I/OAT DMA → huge-page/MMAP 共享内存 → CPU 上重放 optimizer update；约每小时再把维护的状态发送到远端存储池。",
        "快速层为 CPU checkpoint server DRAM；长期层为 remote storage pool。",
        "建议每个 DP ring 监控至少 3 个训练节点；重复镜像流提供 2/3-copy 梯度包冗余，用于无重传的镜像链路恢复。",
        "镜像、解析、梯度拼接与 CPU optimizer update 必须在单个训练 iteration 内结束，GPU 不等待；无法由冗余恢复丢包时，下一 iteration 退化为常规 checkpoint。",
        "内存 checkpoint 在收齐本 iteration 完整梯度并完成 CPU update 后与训练状态一致；远端持久版本只有周期传输完成后才成立。论文未给远端原子发布/commit 协议。",
        "FlowCheck 不拷 GPU snapshot，而是从既有训练网络流重建 checkpoint，并用流内冗余修复镜像丢包。",
        "PDF pp. 2, 5, 8–9, 12–13（Fig. 4、§6.2、§7）。",
        "需要可编程/可配置镜像网络、额外 CPU servers 与 lossless-enough 流量；加密流量、FSDP/复杂优化器适配和远端 commit 不完整。",
        "AegisKV 可借鉴“备份来源可不是 GPU snapshot”：算法 2 可在 Host copy、网络旁路重建、持久落盘之间按资源价格选择。",
        "host_to_persistent_explicit",
    ),
    "TC03": review(
        "runtime_snapshot_or_replication",
        "pinned CPU memory 的 ping-pong pool。",
        "GPU shard → D2H → pinned CPU pool → 多进程序列化 → /dev/shm 共享内存文件 → 上传到 HDFS/NAS/本地盘；HDFS 大文件以定长 sub-file 并发写后做 metadata concat。",
        "生产主后端是 HDFS；也支持 NAS、本地盘和 in-memory backend，并有 SSD→HDD 冷却层。",
        "消除并行布局造成的重复写：复制态仅一份、shard 各写一次；底层存储副本策略不属于 ByteCheckpoint 机制。",
        "D2H、序列化、shared-memory dump、上传完全流水；完整性 barrier 也改为 gRPC tree 并异步执行。",
        "分布式 checkpoint 由多文件组成；全部 rank 文件上传后执行完整性检查/atomic save barrier，才是完整 checkpoint。异步返回与该完整性边界不能混同。",
        "ByteCheckpoint 用并行无关 byte representation、load-time resharding、全异步 I/O pipeline 与后端专用优化统一 LFM checkpoint。",
        "PDF pp. 4–6, 9–10, 12, 19（Fig. 3/5/10、§4.2、App. B）。",
        "生产实现依赖大量框架/存储适配；dataloader state 可达数十 GB；远端 HDFS metadata 与超大规模 barrier 仍需工程优化。",
        "AegisKV 的图应显式画出 D2H-ready 与 distributed-commit 两个事件；α/β 控制的是资源与 drain，不等同于 checkpoint 已可发布。",
        "host_to_persistent_explicit",
    ),
    "TC04": review(
        "format_postprocess_or_empirical_io",
        "N/A：UCP 接收已经写入持久存储的 distributed checkpoint；它不定义训练时 GPU snapshot 的第一落点。",
        "persistent distributed files → CPU 并行 Extract/Union/StripPad → per-parameter atomic checkpoint files → 目标 rank 按 layer 经 CPU memory/H2D 加载。",
        "atomic checkpoint files 所在持久存储。",
        "通过 redundancy-bypassing loading 去掉 DP 复制读；这是 I/O 去重，不是故障副本。",
        "转换仅在需要重配置时 lazy 触发；不是与训练重叠的 snapshot/persist pipeline。",
        "Save 生成 per-parameter atomic files；论文强调结构上的 atomic checkpoint，但未给跨文件崩溃一致性或发布事务。",
        "以 parallelism-independent atomic checkpoint 和 pattern-based reconfiguration 支持 ZeRO/TP/PP/3D 之间恢复。",
        "PDF pp. 5–10, 13–14（Fig. 2、§5.2–§5.4）。",
        "依赖已有基础 checkpoint；转换是重配置成本而非快照优化；“atomic”主要指参数粒度表示，不应误读为持久事务提交。",
        "AegisKV 可把恢复布局兼容性作为算法 2 的可行性约束，但不要把格式原子性当作备份 durability。",
        "not_applicable_live_snapshot",
    ),
    "TC05": review(
        "runtime_snapshot_or_replication",
        "本机与 peer machine 的 Host CPU memory。",
        "GPU model/optimizer shard → dual CPU buffers → D2H、serialization 与 peer send 三阶段重叠；peer shard 保存于 CPU memory。论文同时声明 Host-memory + local SSD 分层，但没有给出 CPU→SSD 的频率和完成协议。",
        "快速层为 local/peer Host DRAM；提及 local SSD tier；刻意避免远端 HDFS。",
        "每个 shard 跨 PP/DP/TP failure group 备份到 peer；ZeRO-only 时退化为邻机备份，以抵抗 over-eviction。",
        "D2H 使用独立 CUDA stream；backup 通信与 F/B、model/optimizer update 重叠。optimizer step 仍等待本 rank 自身 checkpoint save 完成。",
        "恢复选择 D2H 与 serialization 已完成的最新版本；本地 create 有等待边界，peer backup 可继续异步。论文没有定义 local SSD drain/commit。",
        "ByteRobust 将高频 Host checkpoint、跨并行组副本、warm standby、诊断和恢复统一到生产运行时。",
        "PDF pp. 5, 9–10, 13–14（Fig. 8–9、§6.3、实现）。",
        "SSD 层只被概述，缺少 drain 与 atomic publication；跨组策略假设 failure domain 可建模，且占用 2 TB 级 Host memory 与网络空隙。",
        "这是最接近 AegisKV 算法 2/3 的参考：先按 failure domain 选副本，再由 α/β 控制 Host 与 SSD；图上必须把“本 rank safe”与“peer safe”拆开。",
        "host_to_persistent_mentioned_not_specified",
    ),
    "TC06": review(
        "runtime_snapshot_or_replication",
        "可配置：CPU DRAM、local SSD，或另一台机器的 DRAM（RDMA）；论文评测多使用 Host DRAM。",
        "CRIU CPU state 与 GPU state 协调；GPU buffer 通过 soft CoW 或 recopy 并发复制到 backend，恢复采用 soft on-demand restore。",
        "CPU DRAM / local SSD / remote DRAM，取决于 backend。",
        "没有内建 checkpoint replica/erasure code。",
        "checkpoint data copy 与 GPU 执行并发；CoW 得到 t1 状态，recopy 在 t2 短暂停机补拷写集；异步调用仅在上一 checkpoint 未结束时才会阻塞。",
        "CoW 在并发 copy 完成后得到等价于 t1 的一致快照；recopy 在 quiesce+补拷后得到 t2 快照。论文保证执行一致性，但未定义 SSD fsync/跨组件持久 commit。",
        "PhoenixOS 以 validated speculation 构造 GPU soft-CoW/recopy/on-demand restore，实现透明 OS-level C/R。",
        "PDF pp. 4–7, 11–14, 18（Fig. 4、§4–§6、App. A.2）。",
        "间接寻址会使 speculation 失败并需保守处理；不同 backend durability 不同；未提供训练框架级分布式一致性。",
        "AegisKV 可借鉴“安全状态时间点”而非复杂公式：算法 2 输出明确的 t1/t2 可恢复边界，算法 3 再调资源。",
        "configurable_backend_includes_host_and_ssd",
    ),
    "TC07": review(
        "recovery_without_new_checkpoint_path",
        "N/A：ATTNChecker 不创建模型 checkpoint。",
        "在 attention 运算内用 ABFT checksum/重算实时检测并修正 INF/NaN，目标是避免回滚到旧 checkpoint。",
        "N/A。",
        "算法级 checksum 冗余，而非 checkpoint 数据副本。",
        "在线检测与修正位于 attention critical path；没有 snapshot/persist 异步边界。",
        "N/A；修正通过校验通过而成立，不存在 checkpoint commit。",
        "针对 attention 的结构化 ABFT，减少因 INF/NaN 传播造成的训练回滚。",
        "PDF pp. 1–2, 5–10, 12。",
        "保护范围集中在 attention/特定数值错误；不能替代硬件、进程或整机故障的 checkpoint。",
        "AegisKV 应把“重算/校验可覆盖的错误”从备份需求中扣除，作为算法 2 的故障类型过滤器。",
        "not_applicable_no_checkpoint_creation",
    ),
    "TC08": review(
        "recovery_without_new_checkpoint_path",
        "N/A：FT2 是在线 token/关键层容错，不创建训练 checkpoint。",
        "利用 first-token 输入覆盖后续 token 的特性，对关键层做范围保护/纠错；无 GPU→Host→storage 路径。",
        "N/A。",
        "关键层/首 token 复用带来的算法冗余，不是状态副本。",
        "在线执行；没有 checkpoint 异步 drain。",
        "N/A。",
        "First-token-inspired critical-layer protection，用较低开销提高 LLM 输出 token 的故障容忍度。",
        "PDF pp. 1–2, 5–9, 12。",
        "面向输出/算子故障模型，不能恢复 optimizer、进程或节点状态。",
        "算法 1 的 criticality 可结合 FT2 式算子保护概率，避免所有故障都落到同一种 backup 策略。",
        "not_applicable_no_checkpoint_creation",
    ),
    "TC09": review(
        "recovery_without_new_checkpoint_path",
        "N/A：Minder 只检测 faulty machine。",
        "采集监控 metric pattern → 自动检测异常机器 → 交给外部训练恢复系统。",
        "N/A。",
        "无 checkpoint 冗余。",
        "在线监控与检测；无 snapshot/persist。",
        "N/A。",
        "用故障前持续一段时间的可区分监控模式，在大规模训练停止前快速定位机器。",
        "PDF pp. 2, 5–10。",
        "定位后如何替换机器、选 checkpoint、恢复一致性均在论文范围外。",
        "可作为算法 1/2 的 failure-probability 与 failure-domain 输入，而不是备份执行器。",
        "not_applicable_no_checkpoint_creation",
    ),
    "TC10": review(
        "recovery_without_new_checkpoint_path",
        "N/A：Holmes 诊断慢 iteration，不创建 checkpoint。",
        "通信 operator graph + 跨 iteration 分析 → 实时定位性能 irregularity。",
        "N/A。",
        "无 checkpoint 冗余。",
        "在线诊断；无 checkpoint pipeline。",
        "N/A。",
        "把通信算子依赖图用于 silent performance irregularity localization。",
        "PDF pp. 2, 5–10。",
        "面向性能异常而非训练状态持久性；诊断成功不等于可恢复。",
        "其 operator graph 可供算法 1 估计 slack，但不应混入算法 2 的 placement/commit 逻辑。",
        "not_applicable_no_checkpoint_creation",
    ),
    "TC11": review(
        "recovery_without_new_checkpoint_path",
        "N/A：Aegis（该论文）是训练云故障诊断系统。",
        "监控/collective instrumentation → failure localization、fail-slow 与交付前检查 → 外部 restart/recovery。",
        "N/A。",
        "无 checkpoint 数据副本。",
        "运行时诊断，不定义 checkpoint 异步边界。",
        "N/A。",
        "Aegis 从通用诊断演化到面向 AI collective 的生产故障定位。",
        "PDF pp. 2, 5–13。",
        "减少 diagnosis idle time，但状态保存、备份层级与恢复提交由其他系统负责。",
        "可把诊断置信度映射为 AegisKV 的风险输入；注意与本项目 AegisKV 名称相同但机制不同。",
        "not_applicable_no_checkpoint_creation",
    ),
    "TC12": review(
        "recovery_without_new_checkpoint_path",
        "N/A：TrainCheck 检查训练 invariant，不生成 checkpoint。",
        "自动推断 invariant → 每 iteration 主动检查 silent training error → 提供 debug context。",
        "N/A。",
        "无 checkpoint 冗余。",
        "在线检查；无持久化 pipeline。",
        "N/A。",
        "以训练特定 invariant 及 proactive checks 捕获静默软件/配置错误。",
        "PDF pp. 2, 5, 10–12。",
        "不能处理所有硬件/系统故障；检测后的回滚点和状态保护不在范围内。",
        "算法 2 可把“checkpoint 是否可信”作为 admission 条件，避免复制已经污染的状态。",
        "not_applicable_no_checkpoint_creation",
    ),
    "TC13": review(
        "recovery_without_new_checkpoint_path",
        "N/A：Mycroft 做 collective tracing/根因定位。",
        "轻量 instrumentation → flow/data-chunk dependency trace → trigger 后在线 root-cause analysis。",
        "N/A。",
        "无 checkpoint 副本。",
        "追踪按 trigger 开启；无 snapshot/persist。",
        "N/A。",
        "暴露 collective 内部 flow 与 data-chunk dependency，定位 hang/性能异常。",
        "PDF pp. 1–2, 5–10, 13。",
        "解决 observability 而非 state recovery；依赖通信库 instrumentation。",
        "其 failure-domain/dependency 可帮助算法 2 避免把主本与备份放在同一通信 blast radius。",
        "not_applicable_no_checkpoint_creation",
    ),
    "TC14": review(
        "recovery_without_new_checkpoint_path",
        "N/A：TrainVerify 验证 execution plan 等价性。",
        "logical model/plan → symbolic DFG → shape reduction + staged SMT verification。",
        "N/A。",
        "无 checkpoint 冗余。",
        "部署/编译期验证；无运行时 checkpoint pipeline。",
        "N/A。",
        "证明分布式并行 execution plan 与逻辑模型等价，避免错误计划污染训练。",
        "PDF pp. 1–2, 5–10。",
        "不检测运行时硬件故障，也不保存/恢复训练状态。",
        "可以在算法 2 执行 layout change 前做 correctness gate，但不能充当 durability 证明。",
        "not_applicable_no_checkpoint_creation",
    ),
    "TC15": review(
        "format_postprocess_or_empirical_io",
        "N/A（在线 snapshot）：LLMTailor 读取已有本地 checkpoint files，并输出新的本地 composite checkpoint。",
        "多个历史 model/optimizer checkpoint files → CPU multiprocessing 按 layer/parameter group 拆选 → 顺序写出新 model、optimizer shards 与 config。",
        "本地文件系统中的 Frankenstein checkpoint。",
        "跨时间版本保留不同 layer，可用于组合；没有副本/纠删码。",
        "离线 merge 工具，不与训练 D2H 或 storage drain 重叠。",
        "所有 weight、optimizer shard、辅助 layer 和 config 都写完后才可恢复；工具保持 load/write 顺序，但未提供 crash-atomic publication。",
        "用 layer-wise partial checkpoint + merge 探索减少保存体积，支持可继续训练的完整 optimizer state。",
        "PDF pp. 1–7, 9（§4–§5、结论、artifact）。",
        "仅支持本地 checkpoint；规则化 selective saving 可能影响质量；不是生产在线 checkpoint engine。",
        "对 AegisKV 的价值在于“备份粒度可按 layer 选择”，但算法 2 必须增加质量/陈旧性 gate，不能只按 size 最小化。",
        "not_applicable_offline_file_transform",
    ),
    "TC16": review(
        "format_postprocess_or_empirical_io",
        "聚合观测显示现代作业先写 node-local storage；论文未说明该介质是 DRAM 还是 SSD。",
        "训练状态 → node-local storage → 训练恢复后异步 drain 到全局 shared storage。",
        "共享存储平台（基于 VAST telemetry 的观测）。",
        "论文不披露各作业副本策略。",
        "global storage 看到的是训练已恢复后的 drain；用 checkpoint-overlap（drain 时间/间隔）而不是 GPU 峰值来规划带宽。",
        "观测数据没有记录 per-job create/commit 元数据；node-local 完成与 shared-storage durability 不能等同。",
        "从 40 个生产作业、85,000+ checkpoint 反推存储需求，指出共享层带宽通常只需支持低 overlap 的平滑 drain。",
        "PDF pp. 1–2（Methods 与 Preliminary findings）。",
        "2 页 WIP；供应商 telemetry、介质与软件栈不透明；是经验趋势而非可复现 checkpoint 协议。",
        "这直接支持 AegisKV 算法 3 用窗口化带宽利用率调 β，而不是按 GPU 总数静态预留峰值带宽。",
        "local_storage_to_shared_explicit_medium_unspecified",
    ),
    "TC17": review(
        "runtime_snapshot_or_replication",
        "独立 CPU shadow cluster 的 Host DRAM；没有 GPU snapshot D2H。",
        "AllReduce reduced gradients → programmable switch tag/multicast → shadow NIC/CPU bucket → CPU optimizer 更新 Host-memory model/optimizer replica。",
        "原型 checkpoint 位于 shadow Host DRAM；评测用 in-memory nullfs 排除持久存储，未实现磁盘提交。",
        "模型按 DP shard 分布在 shadow nodes；PFC/lossless delivery 保证梯度到达，但没有 shadow-state 的持久副本。",
        "GPU 完全不参与 checkpoint；shadow optimizer 必须在下一 GPU iteration 开始前完成，属于旁路但有 pace deadline。",
        "收齐本 iteration 所有 tagged gradient buckets 并完成 shadow optimizer step 后，该 iteration 的内存 checkpoint 才可恢复；无 disk commit/shadow-cluster failure protocol。",
        "Checkmate 用网络 multicast 复制梯度，在 CPU shadow cluster 每 iteration 重建 checkpoint。",
        "PDF pp. 2–8, 10–14（Fig. 3–5、§4.2–§4.4、讨论）。",
        "依赖 lossless/PFC 与 switch programmability；FSDP/ZeRO 仍是未来工作；没有持久层与 shadow failure 容忍。",
        "可把算法 2 的 placement 扩展为“旁路重建到 Host”，并把网络 port/CPU optimizer deadline 纳入资源约束。",
        "host_only_no_persistent_drain",
    ),
    "TC18": review(
        "runtime_snapshot_or_replication",
        "先到本机 CPU memory，再到 r 个 peer nodes 的 remote CPU memory。",
        "每 iteration 的部分 expert/non-expert 状态 GPU→local CPU；异步复制到 r 个 peer Host；pipeline 边界 activation/gradient logs 留在邻居 Host memory。",
        "distributed in-memory checkpoint（peer Host DRAM），无 disk/remote object store。",
        "每个 sparse snapshot 复制到 r 个 peer；系统保留一个已提交 sparse checkpoint 和一个 in-flight 版本。",
        "每 iteration 小 snapshot 与训练完全重叠；Wsparse 个 iteration 组成一个窗口。log GC 仅在新窗口 checkpoint 已持久复制后发生。",
        "当窗口内全部 sparse snapshots 都在 r 个 peers 上“durably replicated”时，整个 sparse checkpoint 才算 persisted；随后才回收旧版本。",
        "MoEvement 以 sparse operator windows、sparse-to-dense replay 和 upstream logging 同时降低保存与局部恢复成本。",
        "PDF pp. 2–8, 12–13, 20–22（Fig. 5–9、§3.2–§3.6）。",
        "只在 Host memory 提供 durability，peer/common-mode failure 风险仍在；恢复需重放且当前主要针对 MoE；Host 内存增量明显。",
        "算法 2 可直接采用“窗口 commit”：K 内各请求/层逐步备份，K+1 只有在窗口副本齐全后才能回收；算法 3 再根据压力调 α/β。",
        "host_only_no_persistent_drain",
    ),
    "TC19": review(
        "runtime_snapshot_or_replication",
        "checkpoint group 内其他 worker 的 remote CPU memory。",
        "无/低冗余状态的 parameter 或 gradient 经训练网络发给 peer CPU；remote CPU optimizer 每 iteration 应用 gradient，维护可直接恢复的完整状态；另有 background full checkpoint 写 remote persistent storage。",
        "快速层为 peer Host DRAM；灾难恢复层为 remote persistent storage（具体介质未披露）。",
        "按用户 failure-tolerance factor k 分 checkpoint groups，组内互换状态；现存训练 replica 也被计入 redundancy，full persistent checkpoint 覆盖 catastrophic failure。",
        "incremental checkpoint communication 与模型计算重叠；完整 checkpoint 由专用后台线程 non-blocking 保存到持久层。",
        "快速版本在 peer 收到并由 remote CPU optimizer 应用本 iteration 更新后可恢复；完整持久 checkpoint 的跨 rank atomic publication/完成检测未定义。",
        "AdaCheck 先在线检测 tensor redundancy，再以 gradient-based incremental remote updating 只保存必要状态，并辅以后台 full checkpoint。",
        "PDF pp. 4–10, 12–13（Fig. 9–10、§7–§9.5）。",
        "恢复概率依赖 k 与独立故障假设；remote persistent save 的格式、commit 与频率未细化；需要 CPU optimizer 与并行策略适配。",
        "这是 AegisKV 资源管理建模的强参考：把 replica 数 k、Host 容量、网络带宽和 persistent full-checkpoint 周期作为明确资源，而不是堆叠复杂目标函数。",
        "host_to_persistent_explicit_parallel_paths",
    ),
    "TC20": review(
        "runtime_snapshot_or_replication",
        "当前实现为 CPU memory。",
        "checkpoint 前 cudaDeviceSynchronize；GPU data buffers 用 cudaMemcpyAsync 拷到 CPU，GPU control state 由 driver-integrated C/R 保存；增量版本与前一 full checkpoint 在 CPU memory 合并。",
        "CPU DRAM；SSD/remote memory 仅列为 future backend。",
        "无数据副本/纠删码。",
        "dirty-template 的 CPU shadow execution 与 GPU kernel 并行；真正 checkpoint 前仍同步 kernel 并暂停执行，故 memcpyAsync 不等于训练异步 checkpoint。",
        "data buffers 与 control state 都保存、地址映射一致且增量已合并后形成可恢复 checkpoint；没有持久 commit。",
        "GCR 以 control/data separation、driver+interception 混合 C/R 和 CPU shadow dirty tracking 实现快速透明 GPU C/R。",
        "PDF pp. 4–10, 12–14（§4–§6、实现与讨论）。",
        "当前 checkpoint 只在 CPU memory；训练 checkpoint 仍会 stall；增量优化主要围绕 inference KV cache，SSD/remote backend 未实现。",
        "图中要严格区分“异步识别 dirty state”与“同步 create checkpoint”；算法 3 不应把 async API 调用误判为零阻塞。",
        "host_only_no_persistent_drain",
    ),
    "TC21": review(
        "recovery_without_new_checkpoint_path",
        "N/A：TrainMover 不提供新的 checkpoint engine；joiner 可从 leaver/DP replica/remote CPU checkpoint/remote storage 取状态。",
        "expected event：leaver→joiner 一对一 RDMA state transfer；unexpected event：优先从 stayer redundancy/remote CPU checkpoint 取状态，否则从 remote storage checkpoint 恢复。",
        "沿用外部 checkpoint 的 Host 或远端存储介质。",
        "spare GPU、DP redundancy 和 general standby；并非新 checkpoint replica。",
        "sandbox warmup 与两阶段 CCL setup 在后台重叠；最终 membership switchover 与 state readiness 是短暂停机边界。",
        "joiner 状态、通信拓扑和 CCL 都 ready 后才切换角色；外部 checkpoint 的 create/commit 不在论文范围。",
        "TrainMover 把初始化、CCL 建链和角色 warmup 移出 interruption critical path，实现低停机迁移。",
        "PDF pp. 2, 4–10, 15–16。",
        "依赖现有状态来源；异步训练不支持；应用正确性与 checkpoint durability 均交给用户/外部系统。",
        "算法 2 应把“直接迁移/利用现有 replica”作为比新增备份更便宜的动作；算法 3 管理 standby 与网络 headroom。",
        "not_applicable_consumes_external_checkpoint",
    ),
    "TC22": review(
        "runtime_snapshot_or_replication",
        "ByteCheckpoint 的 pinned local Host memory。",
        "每个 RL step：GPU shard → pinned Host memory（短阻塞）→ 异步写 HDFS；检测异常时可预取 HDFS shard 到本地 memory，再恢复 trainer。",
        "HDFS。",
        "rollout replicas/warm standby 提供角色可用性；checkpoint 数据副本由 HDFS 后端负责，论文未披露因子。",
        "只阻塞 trainer 的 GPU→Host；memory→HDFS 非阻塞且与 rollout 继续生成重叠。",
        "Host snapshot 完成不等于 durable；只有 HDFS 写完才可跨机恢复。RobustRL 沿用 ByteCheckpoint，但未单独描述 distributed atomic publication。",
        "RobustRL 以 role-based Detect-Restart-Reconnect 隔离 trainer/rollout failure，并用 per-step checkpoint 保持权重一致。",
        "PDF pp. 5–7, 10, 12–14（§5.1、Fig. 20–21）。",
        "checkpoint 机制来自 ByteCheckpoint 而非论文新贡献；per-step 合理性依赖 RL step 足够长；HDFS failure/commit 细节未评测。",
        "对 AegisKV 的启示是请求/角色 slack 可决定 Host→disk 是否来得及；长 RL step 允许更积极的 β drain。",
        "host_to_persistent_explicit",
    ),
    "TC23": review(
        "recovery_without_new_checkpoint_path",
        "N/A：SDCHUNTER 消费“最新已验证 checkpoint”做 deterministic replay，不创建 checkpoint。",
        "verified checkpoint + 原始 input/communication trace → 两阶段 deterministic replay → 隔离 SDC GPU → 训练从 verified checkpoint 恢复。",
        "沿用外部 checkpoint 存储。",
        "DP replay replicas 用于诊断投票，不是 checkpoint replica。",
        "Phase 1 阻塞在线恢复；Phase 2/硬件确认可离线继续。无 checkpoint drain。",
        "只有 verified checkpoint 才被允许作为恢复源；create/commit 由外部系统负责。",
        "生产 workload 精确 replay 与分层 signature 比较，识别 data-dependent GPU SDC。",
        "PDF pp. 2–3, 5, 8, 10–12。",
        "需要 deterministic training 与可信 reference replica；无法解决所有 replica 同样损坏；checkpoint 生成不在范围。",
        "AegisKV 应给备份附加 validity 状态：未通过 SDC/一致性 gate 的副本即使持久化也不能计入保护度。",
        "not_applicable_consumes_external_checkpoint",
    ),
    "TC24": review(
        "recovery_without_new_checkpoint_path",
        "N/A：AEGIS 是在线 SDC detection framework。",
        "轻量 cSensor 感知可疑 corruption → cVerifier 确认 → 隔离 faulty GPU/触发外部恢复。",
        "N/A。",
        "检测交叉验证，不是 checkpoint 副本。",
        "在线检测；无 snapshot/persist。",
        "N/A。",
        "以 cSensor-cVerifier 两阶段抽象在 0.86% overhead 下发现生产 SDC。",
        "PDF pp. 2, 5–10, 14。",
        "只解决 SDC detection；恢复点、备份位置与持久一致性不在范围。",
        "可把 verifier 信号作为算法 2 的“副本是否可信”与算法 3 的风险突增事件。",
        "not_applicable_no_checkpoint_creation",
    ),
    "TC25": review(
        "recovery_without_new_checkpoint_path",
        "N/A：OpGuard 对两次训练执行做 bitwise alignment/debugging。",
        "operator-boundary fingerprints → schedule-tolerant mapping → 找到首个 bitwise mismatch。",
        "N/A。",
        "对照执行提供 correctness oracle，不是 checkpoint redundancy。",
        "在线/重放调试；无 checkpoint pipeline。",
        "N/A。",
        "以 semantic-stable operator boundary 和 first mismatch 作为生产 LLM debugging primitive。",
        "PDF pp. 2, 4–7, 11–12。",
        "依赖可控制的 benign nondeterminism 和可比较执行；不提供状态恢复。",
        "算法 2 可在备份前记录轻量 fingerprint，使恢复后可快速验证副本，而无需复制更多公式。",
        "not_applicable_no_checkpoint_creation",
    ),
    "TC26": review(
        "recovery_without_new_checkpoint_path",
        "N/A：论文把 checkpoint save/restart 当成模型参数，不实现存储路径。",
        "抽象为 checkpoint period、save time、detect time、restart time，与 tray/block spare、MTBF/MTTR、placement constraint 一起进入 Goodput/CETT 模型。",
        "N/A。",
        "建模 intra-block spare tray 与 inter-block spare；checkpoint redundancy 由外部方案决定。",
        "仅分析同步 recovery；未给具体 async pipeline。",
        "N/A。",
        "用闭式近似、Markov chain 与模拟器做 GPU sparing/repair/placement 的资源管理决策。",
        "PDF pp. 2–7, 10, 13–14。",
        "示例参数不代表真实生产值；checkpoint 路径、数据一致性和异步提交均被抽象掉。",
        "这是把 AegisKV 专利建模为经典资源管理问题的直接参考：最大化 Goodput/CETT，决策变量用 Host 容量、带宽、spare、failure domain，复杂理论退到离线配置器。",
        "not_applicable_analytical_model",
    ),
    "TC27": review(
        "recovery_without_new_checkpoint_path",
        "N/A：EROICA 是在线性能 troubleshooting system。",
        "全机在线 profiling + differential observability → 定位软硬件性能根因。",
        "N/A。",
        "无 checkpoint replica。",
        "在线 profiling；无 checkpoint pipeline。",
        "N/A。",
        "在约 100,000 GPU 生产集群做细粒度、全覆盖性能诊断。",
        "PDF pp. 2–8。",
        "诊断性能问题，不创建或恢复训练 state。",
        "其在线 profile 可给算法 3 提供 Host/网络/存储实际服务率，替代纯理论带宽估计。",
        "not_applicable_no_checkpoint_creation",
    ),
    "TC28": review(
        "format_postprocess_or_empirical_io",
        "N/A（save path）：输入 checkpoint 已在 persistent storage。",
        "remote/cross-DC checkpoint namespace → 预测复制到目标 DC；启动前把 hot metadata/replicated-tensor files 从默认 3 replicas 扩到约 128 ephemeral replicas → ranks 并发加载。",
        "分布式持久存储节点；TTL 到期自动回收临时副本。",
        "默认 3 副本；按 world size/access fan-out 对热点文件 proactive replica expansion。",
        "复制在 recovery/startup 前完成；真正 checkpoint loading 是所有 ranks 等待最慢者的同步 barrier。",
        "replica readiness 在 recovery 开始前成立；论文不定义 checkpoint save/commit，仅定义 load 前的副本准备。",
        "预测跨 DC checkpoint replication、热点文件预复制与 storage-side transform offload。",
        "PDF pp. 2–4, 8–10, 13–14。",
        "只优化读取/数据管线；依赖可预测 access pattern 与 storage namespace；不会改善初始 checkpoint commit。",
        "算法 2 不应只考虑写入：恢复 fan-out 会反向决定 β 和 storage replica；算法 3 可在 K+1 前预扩热点。",
        "not_applicable_load_only",
    ),
    "TC29": review(
        "runtime_snapshot_or_replication",
        "模型状态本来就位于 follower PS 的 CPU DRAM；论文未披露异步 checkpoint 的第一持久落点。",
        "每个 outer optimizer step 后，distributed follower PS 从其 CPU-resident shard 异步保存 snapshot；leader 只保存 <100 KB metadata；仅保留最近几个版本。",
        "未说明（只写 checkpoint/snapshot storage）。",
        "follower PS 按 layer 分片而非复制；leader-follower 控制面可重启，但 checkpoint 数据副本因子未披露。",
        "follower checkpoint 与 control/data flow 重叠；leader failure 时 inner training non-blocking 继续、跳过 outer update。",
        "leader 用 latest checkpoint 重启 failed follower；论文未给 storage durability、跨 followers atomic point 或 publish protocol。",
        "Di-PS 以 CPU parameter-server 全局状态、异步 outer optimizer 与 follower snapshot 实现跨集群韧性。",
        "PDF pp. 6–9（Fig. 4、§4.1、§4.3）。",
        "destination/commit 语义不透明；distributed followers 的一致 cut 未描述；它是 cross-cluster optimizer 系统而非通用 training checkpoint engine。",
        "提醒 AegisKV：CPU-native control state 与 GPU state 不能套同一 D2H 模型；算法 2 需要按 state owner 区分路径。",
        "cpu_native_source_destination_unspecified",
    ),
    "TC30": review(
        "recovery_without_new_checkpoint_path",
        "N/A：FLARE 是全栈 anomaly diagnostics。",
        "轻量 tracing daemon → backend-extensible trace → 自动 diagnostic engine 定位 performance regression/divergence。",
        "N/A。",
        "无 checkpoint redundancy。",
        "在线 tracing；无 snapshot/persist。",
        "N/A。",
        "覆盖训练全栈的 anomaly diagnostics，已在 6,000 GPUs 部署。",
        "PDF pp. 2–8, 11。",
        "诊断覆盖不等于故障恢复；checkpoint create/load 都不在范围。",
        "可把诊断输出变成算法 3 的事件输入，让 α/β 对故障与资源退化做窗口化调整。",
        "not_applicable_no_checkpoint_creation",
    ),
}


MANIFEST_FIELDS = [
    "id",
    "year",
    "venue",
    "venue_scope",
    "title",
    "venue_url",
    "pdf_url",
    "pdf_source",
    "pdf_filename",
    "directness",
    "system_group",
    "pdf_magic",
    "bytes",
    "pages",
    "sha256",
    "first_destination",
    "staging_path",
    "final_medium",
    "redundancy",
    "async_boundary",
    "commit_semantics",
    "key_mechanism",
    "evidence_pages",
    "limitations",
    "aegiskv_insight",
    "host_to_storage_class",
]


def load_download_rows() -> list[dict[str, str]]:
    with (ROOT / "download_validation.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 30, len(rows)
    assert set(REVIEWS) == {row["id"] for row in rows}
    return rows


def merged_rows() -> list[dict[str, str]]:
    result = []
    for row in load_download_rows():
        semantic = REVIEWS[row["id"]]
        merged = dict(row)
        merged["pdf_filename"] = merged.pop("filename")
        merged.update(semantic)
        result.append(merged)
    return result


def write_manifest(rows: list[dict[str, str]]) -> None:
    with (ROOT / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def class_label(name: str) -> str:
    labels = {
        "runtime_snapshot_or_replication": "运行时 snapshot / state replication",
        "format_postprocess_or_empirical_io": "格式、后处理、读取或经验 I/O",
        "recovery_without_new_checkpoint_path": "恢复/诊断相关，但不定义新 checkpoint 写路径",
    }
    return labels[name]


def write_review(rows: list[dict[str, str]]) -> None:
    venue_counts = Counter(row["venue_scope"] for row in rows)
    group_counts = Counter(row["system_group"] for row in rows)
    host_counts = Counter(row["host_to_storage_class"] for row in rows)

    lines = [
        "# 2025–2026 LLM Training Checkpoint / Fault-Tolerance Paper Review",
        "",
        "本目录固定收录 30 篇论文。PDF 已逐文件验证 `%PDF-` magic、可解析页数、SHA-256，并确认无重复哈希；下载级证据见 `download_validation.csv`，合并后的机器可读索引见 `manifest.csv`。",
        "",
        "## 口径与结论",
        "",
        f"- 发表口径：main conference **{venue_counts['main']}** 篇，peer-reviewed workshop **{venue_counts['peer-reviewed workshop']}** 篇，work-in-progress **{venue_counts['work-in-progress']}** 篇。后两类在每篇条目中显式标注，不冒充顶会主会论文。",
        f"- 语义分组：运行时 snapshot / state replication **{group_counts['runtime_snapshot_or_replication']}** 篇；格式、后处理、读取或经验 I/O **{group_counts['format_postprocess_or_empirical_io']}** 篇；只做恢复、诊断、校验或资源规划且不定义新写路径 **{group_counts['recovery_without_new_checkpoint_path']}** 篇。",
        "- 对 11 篇“运行时 snapshot / state replication”论文逐一看第一落点：**9/11 明确把 Host DRAM（本机、peer 或独立 CPU server）作为第一备份/暂存层**；PhoenixOS 是可配置 DRAM/SSD/remote-DRAM backend；Di-PS 的状态原生位于 CPU PS，但未披露 snapshot destination。",
        "- 在这 9 篇明确 Host-first 的论文中，**5/9 给出 Host→持久层的明确路径**（PCcheck、FlowCheck、ByteCheckpoint、AdaCheck、RobustRL）；**1/9 只提及 Host+SSD 分层却未给 drain/commit**（ByteRobust）；**3/9 原型停在 Host memory**（Checkmate、MoEvement、GCR）。因此严谨结论是：**Host DRAM 作为快速第一层是主流；Host→disk/remote persistent storage 不是所有系统的共同终点，更不是每篇都给出可验证 commit 语义。**",
        "- 另有 85,000-checkpoint WIP 观察到生产作业常先写 node-local storage，再异步 drain 到 shared storage；它没有披露 node-local 介质，因此不能把该结果自动计入 Host-DRAM-first。",
        "- `async` 必须拆成至少三种：snapshot/create 异步、Host/peer replication 异步、persistent drain 异步；只有最后的 distributed integrity / fence / msync / replica-window completion 才能作为相应层级的 commit。",
        "",
        "Host→storage 分类的原始计数（用于复核，不代表质量排序）：",
        "",
    ]
    for name, count in sorted(host_counts.items()):
        lines.append(f"- `{name}`: {count}")

    lines += [
        "",
        "## 逐篇审读",
        "",
    ]

    for row in rows:
        lines += [
            f"### {row['id']} — {row['title']}",
            "",
            f"- 年份 / venue：{row['year']} / {row['venue']}（`{row['venue_scope']}`）",
            f"- 范围：`{row['directness']}`；{class_label(row['system_group'])}",
            f"- 页面 / SHA-256：{row['pages']} pages / `{row['sha256']}`",
            f"- 链接：[venue]({row['venue_url']})；[PDF source]({row['pdf_url']})；本地 `{row['pdf_filename']}`",
            f"- checkpoint 第一落点：{row['first_destination']}",
            f"- staging path：{row['staging_path']}",
            f"- 最终介质：{row['final_medium']}",
            f"- redundancy：{row['redundancy']}",
            f"- async boundary：{row['async_boundary']}",
            f"- commit semantics：{row['commit_semantics']}",
            f"- 核心机制：{row['key_mechanism']}",
            f"- 对 AegisKV 的启示：{row['aegiskv_insight']}",
            f"- 证据页：{row['evidence_pages']}",
            f"- 局限：{row['limitations']}",
            "",
        ]

    lines += [
        "## 文件级审计说明",
        "",
        "- `download_validation.csv` 保存原始下载 URL、来源类型、bytes、pages、SHA-256 和 PDF magic。",
        "- `manifest.csv` 在其基础上加入逐篇语义字段。所有 `N/A` 都表示论文未定义该机制，而不是审读遗漏。",
        "- PDF 页码均为文件中的 1-based PDF page number，便于直接跳转核对；它可能与论文印刷页码不同。",
        "- 2026 论文均来自会议官方 proceedings 页面或其官方 PDF；2025 的作者 accepted/VOR PDF 在 `pdf_source` 中单独标明。",
        "",
    ]
    (ROOT / "per-paper-review.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = merged_rows()
    write_manifest(rows)
    write_review(rows)
    print(f"wrote manifest.csv and per-paper-review.md for {len(rows)} papers")


if __name__ == "__main__":
    main()
