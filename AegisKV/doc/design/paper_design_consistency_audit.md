# AegisKV 论文设计一致性审计与闭环

> 日期：2026-08-12  
> 原审计对象：`paper/main.tex` 与 `paper/sections/03-system-design.tex`  
> 设计权威：`design/current_design.md` 与 `design/theory_appendix.md`
>
> **当前状态：已闭环。** `paper/main.tex` 已改写摘要并切换到
> `paper/sections/03-system-design-v2.tex`；旧 `03-system-design.tex` 仅保留为
> 历史版本。下列“旧正文”冲突表用于记录为何不能回退到旧设计。

## 1. 结论

原 System Design 正文不能作为最新 AegisKV 设计继续局部润色。它描述的是一套旧系统：固定完整 backup DP、safe frontier/manifest、在线 failure-scenario MIP，以及用嵌套优化调整 HBM envelope。最新设计则是四组件协作、连续已提交 prefix、固定 Router recovery assignment、Slack deadline 驱动的 KV action/EDF，以及只控制 residual protection 的 AIMD。两者的算法、状态 owner、故障协议和理论边界均不同。

该审计决定已经执行：旧稿保留为历史版本，新增的
`03-system-design-v2.tex` 已经过独立编译和页面检查，并由 `main.tex`
引用。迁移不是术语替换，而是算法、状态 owner、协议和证据边界的重写。
2026-08-12 最新构建共 9 页，References 从第 8 页开始；System Design 位于
第 3--7 页并在第 7 页右栏上部结束，约 4.5--5 页。第 7 页其余空白来自当前主动
`clearpage`，不是缺页或 clipping。当前审计只判断 System Design/Algorithm Design
能否自洽通过 reviewer；workload coverage、性能和端到端收益留给后续 Evaluation，
也不能把预留页面当作已经完成的系统证据。

Background/Motivation 的问题较小，叙事骨架和主要现象可以保留，但仍需四处定点修订：把预测点估计改成 fail-closed 的 `BOUNDED/UNKNOWN` contract；删除 mixed-tier manifest 与互斥 placement 的前向引用；把预冻结 complete backup DP 改为 Router 指定的 compatible recovery target；把 “solver optimality” requirement 改为 online heuristic 加离线 oracle。它不需要整体重写。

## 2. 致命冲突

| 旧正文主张 | 最新设计 | 为什么必须重写 |
|---|---|---|
| admission 时冻结一个完整 backup DP `b(r)` | Router/Conductor 先发布 standby，故障后给每个请求固定兼容 recovery target；Controller 不负责 assignment | 固定完整 replica 会把容量假设、故障域和恢复顺序全部写死，也掩盖 Router 与 Controller 的职责边界 |
| logical master 同时拥有 placement、commit、route/manifest | Router 拥有 route/health/output log；Controller 拥有 Slack/action；Store Master 拥有 Store object/lease；Worker 拥有 HBM pages | 旧稿把 Mooncake-style Store Master 与 Aegis policy Controller 混成一个新中心化 master，既难实施又形成单点 |
| safe frontier 与 manifest 是 Aegis 新抽象 | 使用 Store/Worker 已有的 complete object metadata；Aegis 只要求 continuous committed prefix、group commit 和 lifecycle fence | 旧术语夸大 substrate novelty，并与简洁的现有系统映射冲突 |
| 每个 chunk 在 HBM/DRAM/SSD/Recompute 间做互斥 tier placement | DRAM/SSD 是已有 Store cache/offload；Aegis 新增的是 quota-backed tier lease、peer-HBM shadow，以及 retain/extend/accelerate/release action | 旧稿把已有多级存储当成新的 backup placement 算法，容易被 related-work reviewer 直接否定 |
| Algorithm 2 在线枚举未来 failure trace、预测 ACK/WCT，并求 MIP/CVaR | 在线 Algorithm 2 只重放当前的单-domain “fail now” contingencies，以无参数的 `(worst deadline miss, total exposure)` 排序 action，并使用 multi-resource feasibility 与 per-target EDF；完整 `J_d` 只作离线 oracle | 旧算法依赖不可可靠观测的未来和大量 surrogate，违背最新的 simplicity/causality 目标 |
| Algorithm 3 通过嵌套求解 `J_k^*(alpha)` 选 envelope | Algorithm 3 用 TCP-style AIMD 调 `alpha_g` 和逐资源 `copy_rate_r`；`F_serve/F_recover` 是硬契约，不由 AIMD 改写 | 旧稿既不是选定的 AIMD，也无法给出在线常数开销和故障时快速退让的 story |
| 故障后整请求在预冻结 backup DP 上恢复 | Router fence/改 route version，Controller 只调度 ready recovery jobs；peer shadow 需先 `ADOPTED`，Store/recompute 可组合恢复连续 prefix | 旧恢复协议没有 release/deadline 分离、dynamic EDF、shadow adoption 或 UNKNOWN fallback |
| abstract 声称 solver-certified finite causal snapshot | 核心在线路径不依赖 solver 最优；精确/MILP 只用于离线 oracle 和 regret | abstract 当前总结了已经废弃的核心贡献，必须最后随 v2 一起重写 |

## 3. 不需要性能实验的内容

这些结论应放在正文的简短命题/反例与附录证明中。不要为它们重复画跨模型性能图。

| 主张 | 充分证据 |
|---|---|
| 相同 causal history 在无 continuation bound 时不能推出正 hard Slack | T1 indistinguishability counterexample |
| `s_i^B=[L_{-i}-U_i]^+` 是保守 Slack lower bound；故障 cutoff 只冻结 `[deadline_at-F]^+`，过期 ticket clamp 为零而不是产生 age bias，target release 不充值 | T1 单调性、barrier-expiry 与 cutoff-normalization proof |
| `L_{-i}<U_i` 的 zero ticket 不具有“赶在 sibling 前”的 barrier 含义 | T1 分情况证明；cutoff 后以 `D_i^F=F` 和完整 path shift `P_i<=C_i-F` 给出 harm bound，不能误写成无正部的 `t_0+L_{-i}-U_i` |
| hard duration witness 必须覆盖故障后最快 sibling 与最慢受约束 target continuation，不能只来自健康配置均值 | T1 counterfactual quantifier audit；实验只测 tightness/fallback |
| Algorithm 1 的 hard bounds 不能停留在“应用声明”口号 | node interval 由 mandatory work lower/upper、physical service ceiling、reserved service floor 与 bounded tool latency 构造；共享资源 floors 必须联合准入且 queue bounds 可组合，缺 endpoint 或容量不可行就 `UNKNOWN` |
| weakening graph/service event 不能先于 hard-ticket fence 生效 | T1 distributed-order counterexample；这是 invalidate-before-effect 协议前提，不是 stale-rate 实验 |
| 一个裸 workflow SLO 不能把 `UNKNOWN` recovery Slack 升级为 `BOUNDED` | T1 两 continuation 具有相同 SLO、不同 post-recovery work 的反例；只有 finite hard `U_post` 才给出独立的 `T_w-U_post` SLO-feasibility deadline |
| `Delta H=[p_0-s]^+-[p_1-s]^+` 的三段结构 | T2 代数恒等式 |
| `BOUNDED` Slack 得到的是 harm 上界，而两个上界之差不是 realized benefit 下界 | T2 单调性与 `s^B=0,s=10,p_0=5,p_1=0` 反例；在线目标解释为 deadline-risk reduction |
| exact Slack/service 也不自动让早期 absolute `C-D` 等于 realized WCT | T2 common-origin 条件与两 branch 时间轴反例；部署统一称 conservative deadline risk |
| Slack-deadline action 与 size/load 或 binary criticality 不等价 | T2 构造反例；实验只测该差异在真实 workload 中是否常见 |
| frontier antichain 用 `max` 聚合，causal chain 不能直接用 `max` | T2 lemma 与 sequential counterexample |
| 未 group-commit action 的 recovery gain 为 0 | 状态可见性 lemma |
| admission 的 `S\oplus a` 只表示 post-commit action state，commit 前 realized value 为 0；`tau_a` 作为 fail-now origin，只把 positive ticket remainder 消耗到零 | visibility lemma、horizon cutoff-normalization；实验只量化 lead-time opportunity/waste |
| current snapshot + copy completion bound 不能识别未来 job 是否仍存活或 tail 如何增长，因此不能给出精确 post-commit action value | T2 两 continuation 反例；source/route/object witness 只证明 action 可提交；无 job-liveness 与 prefix/service contract 时标 myopic snapshot score，完整 contract 则取允许状态上的最小 risk reduction |
| soft Store priority ACK 不能证明未来 failure-time medium | T2 两 continuation 反例；hard score 只接受可拒绝、quota-backed、覆盖 horizon 的 tier lease；无需性能实验 |
| Controller 不能把 Master absolute expiry 当作本地 deadline | T2 tier-lease clock lemma：只用本地 send/receive 与 TTL，`e_C=t_send+T-epsilon_clk`；RTT 已耗尽安全 interval 就不进入 `S` |
| Master failover 不能静默缩短已 ACK lease | T2 failover corollary：ACTIVE placement/quota/record 先 durable，terminal state 先于 demotion，新 epoch fencing，takeover 后保守重启完整 TTL |
| recovery read lease 在 Controller/Master failover 后不能丢 ref 或只重建子集 | 新 Controller 只 rebind 精确 group/member set；Master HA 复制 refcounts或给已 ACK keys bounded takeover grace，缺失则该 Store edge退出 theorem mode |
| Controller `issuer_epoch` 与 Store Master `authority_epoch` 不能混用 | 前者 fence policy mutation，后者 fence Store leader；commit `creation_epoch` 又只标识 immutable content lifecycle |
| Store protection capacity 必须按 Master actual physical bytes 计费 | Master 而非 Controller 选择/返回 replica-set digest；commit record 与全部 required members 按目标层实际 bytes 保护/计费；promotion 的 PROCESSING DRAM 与 SSD source pin 分别入账；只有相同返回 set 的 PENDING/ACTIVE references 共享，最强 `min_medium` 决定 reservation |
| peer/Store placement 不能与 source failure domain 共损 | commit eligibility 显式验证 placement domain；peer action 还需 Router standby-target/route capability；route-version bump 直接 fence，Controller 异步清理，orphan bytes 到 terminal reclaim 前继续计费 |
| `rho` 在 zero budget/zero incremental demand 时不能除零 | positive demand + zero budget 直接 infeasible；全零 demand 的 positive-gain bundle 先接纳，nonpositive 忽略；traffic 用 bytes / `(rate*window)` |
| Algorithm 2 候选若枚举所有 prefix intervals 会不可扩展 | next/furthest sealed、deadline/tier crossing 与 geometric endpoints；dependency closure 原子成 bundle；明确无 pruning optimality claim，decision timeout fail-safe |
| “least risk-loss cover” 暗示在线又精确求一个组合优化 | 明确定义逐轮 `Loss/cover` 的 deterministic greedy victim cover；`A,V,K` 与 endpoints 全部 capped，保守复杂度 `O(K A V^2 T_eval)`；timeout 保留 committed state |
| `S-V+a` 把满容量下的 reclaim/copy 假装成物理原子 swap | terminal proposal 与 actuation 分离；跨 owner 用 TTL PREPARE/ABORT；make-before-break 优先，break-before-make 必须逐 phase 不劣于同刻 baseline，partial/abort state 显式评分 |
| `F_recover` 若被健康流量借用，故障时不能瞬时凭空收回 | 只允许 bounded terminal、generation-fenced loan；revoke 延迟计入 `R_j/B_g`，否则保持物理/队列隔离 |
| AIMD fast/slow loop 不能用一个混合信号同时误调两个 actuator | transport/latency signal 只调 `copyRate_r`，memory signal 只调 `alpha_g`；同时 overload 才独立触发两次更新 |
| `B_peer` 不能被误读为增量 free bytes | 它是 total shadow occupancy cap；新 reservation 同时检查 `U_peer+q<=B_peer` 与原子 capacity ledger |
| D2D adoption 不能借用为正常 primary growth 保留的 safety reserve | fresh destination 计入 `U_primary`，但 begin 只在完整 `Reserve^*` 之外仍有 `q` bytes 物理 headroom 时提交；pending 阶段关闭 peer admission，否则选 zero-copy/Store/recompute |
| 增加 failure-valid state 不应使 Recovery Planner 的 service bound 变差 | T2 plan-set inclusion + null-option lemma；无需性能实验 |
| token-boundary DAG shortest path 只对串行 continuous-prefix grammar 精确，不能表达一般 token/layer/GPU overlap | T2 两个 5 秒并行 operations 的 5 秒 makespan 与 10 秒 additive path 反例；无需 GPU 实验即可否定一般等价性 |
| pipelined mode 不能把多个 jobs 的独立最优 makespan 拼成 completion vector | T2 两 job 共享 capacity-1 compute/I/O 的 capacity 反例；必须由 scenario-level joint planner 返回 `{C_j}`，无需实验即可证明独立拼接不成立 |
| 理想单机抢占 EDF 最小化 maximum lateness | 引用 Horn；无需重新用实验“证明” |
| live deadline/service version 改变后不能声称相对 future-aware full-history optimum 仍最优 | T4 两 job tightening 反例；新版本只从当前 remaining-work state 与相同 executed prefix 开始 suffix comparison |
| EDF 只优化 lexicographic risk 的第一项，不保证 total overrun 最优 | T4 三 job 反例；第二项只在固定 dispatch 下评估并用 oracle 报 regret |
| exact protection selection 即使在独立 action、单资源下也为 NP-hard | T2 从 0--1 knapsack 的归约；不需要性能实验，oracle replay 只量化当前 greedy 的实际 gap |
| 无 hazard input 的 minimax 不能同时声称 expected-loss optimality 或 tenant fairness | T2 两域 probability 反例；core 只是 canonical risk pool 内的 tail policy，fairness 来自硬 quota |
| 严格单 lane chunked EDF 与理想 EDF 相差小于一个 bounded quantum | T4 critical-block proof；实现只需验证 premises |
| otherwise-idle `DEFERRED` work 不等于零 foreground interference | T4 arrival counterexample；单 outstanding isolated lane 只保证至多一个 `<B_g` background block，无需性能实验 |
| `U_primary+U_peer+Reserve<=C_KV` 在原子 ledger 下保持 | T5 状态转移归纳 |
| AIMD 降低 peer cap 不等于已存在 shadow 瞬时消失 | T5 desired-cap/occupancy 分离；DRAINING 只在 terminal ACK 后减账，pin 可造成显式 overdraft |
| 经典实时/DAG 系统已存在 Slack-aware checkpoint 与 deadline recovery，因此这组泛化概念不能作为 Aegis novelty | Melhem--Mossé--Elnozahy 2004 与 Acharya--Mahapatra 2008 的目标/模型对照；这是 related-work/claim-scope 分析，不需要重做性能实验 |
| peer shadow 在 adoption 前不可作为普通 GPU hit 路由 | page-table/route state invariant |
| 单 rank 已写 request mapping 不等于多 rank request 可以运行 | quarantine + all-rank `ADOPT_COMMIT` route-permit lemma；分布式原子性在线性化的 scheduler/router admission，而非假设 page-table 物理同时写入 |
| recovery KV 必须截断到 durable continuation boundary，缺口 token 只 silent replay 一次 | T3 明确半开边界 `B=P+n`：完整 KV end `K<=B`，prompt `[K,P)` 与 accepted output `max(1,K-P+1)..n` 恰好覆盖 `[K,B)`；无需吞吐实验 |
| token 与 next continuation-state 不能分两次对外可见 | T3 两个 crash-cut 反例：先外发 token 会留下不可继续的 accepted prefix，先发布 state 会缺少 replay token；可重建 state 绑定规则，不可重建 blob 先 durable，Router WAL 再原子引用并提交 token；理论必要性无需性能实验 |
| Controller failover 后旧 commit 仍可读、旧 leader 不能修改 | creation epoch / issuer epoch 分离的 single-writer invariant |
| KV compatibility 与请求 continuation 完整性不是同一件事 | adapter/multimodal 与 RNG/grammar 两个反例；分别要求完整 compatibility/prefix identity 与 durable continuation-state，无需性能实验 |
| digest/reference 不等于 side-input payload 可恢复，single sequence 也不等于完整 beam state | T3 source-local payload 与 competing-hypothesis 反例；E1 要求 failure-surviving canonical object，并收窄为一条 authoritative sequence |

## 4. 只需机制正确性验证的内容

这些不是大规模性能实验，但仅有纸面证明也不足以说明实现遵守协议。

| 机制 | 最小验证 |
|---|---|
| group commit、epoch fence、连续 prefix | state-machine property tests 与乱序/迟到 ACK fault injection；共享 group identity、rank-local checksum/generation、creation/issuer epoch 分离 |
| Store 多 object 原子可见 | 用普通 object put 写 members、marker-last commit record 才发布 recovery visibility；`AcquireGroupReadLease` 在一个 Master transaction 内验证并 ref-pin record 和全部 members，partial acquisition rollback；partial `BatchPutEnd`、missing member/checksum、marker/member eviction 与 lease-vs-evict tests。group ID 只作为 routing/lease/eviction substrate，不能替代 commit record |
| commit/reclaim/failure snapshot 并发 | target-local `FreezeAndPin` linearization、partial group pin rollback、Store bounded read-lease tests |
| sealed immutable chunk | 写入期间 snapshot/COW guard 检查；不满足时强制重算 tail |
| HBM shadow adoption | private page 不得进入 request/radix namespace；source 与 destination generation 分别验证；partial-rank `ADOPT_PREPARED` rollback；route permit 必须先于 scheduler enqueue/GPU event。P0 D2D 路径计双份瞬时 occupancy并在失败时 `is_insert=false` 释放 fresh pages，P1 zero-copy 再检查原子 ownership transfer |
| Recovery Planner | selected-set 重跑、旧 plan/schedule null option、operation DAG terminal ACK/bound；pipelined mode 检查 joint schedule resource feasibility，串行 reference 另检查 group continuity、commit/pin/lease/layout/failure-domain eligibility 与 lifetime 分离 |
| output continuation | Router token/next-state atomic commit 必须先于 stream publish；canonical input records/previous-record/token/position/state hash-chain、torn tail、outbox ordering、worker fence、hard-kill/speculative decode 后 durable-boundary truncation、single silent replay 与 tool-call idempotency检查 |
| request semantic continuation | compatibility/prefix identity 版本拒绝、canonical side-input object resolve/auth；标准 single-sequence SGLang penalty/grammar/stop 由 accepted-token log 重建，seeded mode恢复绝对 sampling position；unsupported custom processor、multi-hypothesis 与 draft state fail closed |
| Store tier lease | immutable group/version/digest、Master actual-byte check、unique-byte aggregate quota、PENDING 零价值、最低驻留层、durable placement ACK、本地安全 expiry、leader change、lease expiry/release-before-demotion 与旧 epoch 拒绝；soft priority 不得进入 deterministic `S` |
| theorem-mode EDF | 检查一个 outstanding quantum、terminal ACK bound、无 cross-job overlap、setup 全计入 |
| dynamic EDF input update | route/due/service version 原子 re-key/rebind；旧 service contract 覆盖已发 quantum 到 terminal，否则退出 theorem mode |
| pipelined deadline-first mode | 每个 contended resource 的 ready/deferred precedence 与 due-date propagation；只验证机制，不宣称全局 EDF theorem，性能用 joint-resource oracle 比较 |
| 资源账本 | allocator mutation property tests；容量缩减和不可回收 pin 边界 |
| hard-ticket lifetime | witness contract 覆盖 upper completion horizon；weakening event 在 Controller downgrade/ACK 后才生效 |
| recovery completion boundary | terminal ACK 必须覆盖到同一 synchronization boundary 的全部 failure-only recovery/catch-up；否则关闭 `C_i-D_i` harm bound |

当前仓库已经把其中一部分压缩成独立的 metadata protocol model：
`aegis_recovery_protocol.py` 的 50 个 targeted tests 检查 all-rank commit、
共享 group identity 与合法的 rank-local checksum/source/destination generation 差异、
partial `FreezeAndPin`/adoption rollback、commit 前不可路由、empty pin 不生成 permit、
leader-epoch fencing/rebind、pin-vs-reclaim、Store exact read-lease rebind、
Store marker-last visibility、partial per-object completion、member checksum/generation、
longest continuous leased prefix、missing-member fail-closed 与 lease-vs-evict，
durable-boundary truncation、single replay、token/next-state atomic commit、
input-rooted hash-chain/torn-tail rejection、worker fence、ordered outbox，以及
failure-surviving canonical input、compatibility/prefix identity、versioned continuation-state、
single-authoritative-sequence 与 draft-state discard 的 fail-closed 边界。它们属于本节的
**模型符合性证据**，不是 actual-byte allocator/WAL/Store 集成，也不替代少量
hard-kill correctness tests。

`aegis_store_lease.py` 的 16 个 tests 另外闭合 Store protection contract：Master
actual-byte authority、PENDING reservation/zero value、promotion terminal ACK、同一对象
同 placement unique-byte/strongest-tier aggregation、不同 placement 分账、source-domain
survival、expiry/release-before-demotion、本地 TTL 安全点、
renewal、failover TTL preservation 与旧 epoch fencing。它复用现有 Store
quota/promotion/eviction substrate 的语义，但不是 runtime adapter 已完成的证据。

Algorithm 2 的 selection bookkeeping 也已有独立 reference model：
`aegis_prefix_recovery.py` 只检查 serial mixed-tier continuous-prefix shortest path，
`aegis_recovery_laxity.py` 检查 fixed-target dynamic EDF，
`aegis_protection_portfolio.py` 已把 action-gated prefix edges 直接接入每个
selected set 的 shortest-path 重算，并检查跨 domain robust pair、typed
multi-resource caps、prefix closure、victim exchange、absolute-deadline
commit-horizon gain、break-before-make phase rejection、dynamic due-version suffix re-key 与 static exhaustive certificate。当前三文件合计 47 个
targeted tests。它们不表达 CacheFlow 式 token/layer/GPU overlap 或 batch restoration。static oracle 明确拒绝 nonzero copy lead time，避免把 completed
snapshot certificate 冒充 time-expanded issuance optimality；这些仍只是机制/语义
符合性，不通过 G0/G1、G3 或 G4。

Algorithm 1 的 hard-bound path 不再只有 enum/singleton zero：
`aegis_bounded_slack.py` 已实现 complete residual DAG 或 closed mandatory-
AND-join region 上的 `L_without/U_through`，检查全部 nodes 可达 barrier，
保留 original source/sink，并对 missing/stale/过早到期 witness 整图 fail closed；
typed work/service envelope 缺正 reserved rate 或单位不同时拒绝 finite upper。
14 个 targeted tests（含 duration-bound endpoint 穷举和 witness-horizon 边界）只说明
公式实现安全；live workflow adapter、service-envelope enforcement、invalidate-before-effect
和 positive-bound prevalence 仍未完成，所以 G1 状态不变。

G1 的 evidence plumbing 现在也分成两层：v3 scorer 只产出不带 gate claim 的
point estimates；`assess_bounded_opportunity.py` 只有在 frozen formal collection
contract 完全一致且零 schema rejection 时，才用 workflow-unit Wilson interval 与
workflow-cluster bootstrap 判断 coverage 子门槛。它永远不把 coverage pass 升级成
完整 G1，因为 v3 尚未记录 protected/unprotected crossing 与 EDF dispatch change；
这些仍需要另一份冻结的 decision-study contract，而不是从同一 point estimate 推断。

`aegis_peer_reserve.py` 的 16 个 tests 进一步把 AIMD desired cap 与 actual occupancy
拆开：cap decrease 只触发 `DRAINING` 和 admission stop，只有 terminal reclaim/adoption
才减少已计费 bytes，并显式报告 cap overdraft。D2D pending 必须同时是已计费 source
shadow 与 fresh primary destination 的子集，marker 本身不形成第三份 charge；begin 若
需要借用 primary safety reserve 就原子拒绝。它仍依赖实际 allocator 证明某 bytes 确实
unpinned/reclaimable，不能由 reference ledger 代替。

peer-HBM live-code mapping 也已经从“泛泛复用 Mooncake”收敛到可执行边界。当前
SGLang Mooncake PD decode 在 target 先分配真实 KV pages并写 request mapping，再把 page
indices、已注册 buffer pointers 和 session 交给 source；source 用同步 batch write 直写
HBM，decode 以 metadata gate + Gloo `MIN` all-reduce 统一成功，失败时
`release_kv_cache(..., is_insert=false)`。Aegis 可以复用 Transfer Engine、registered
buffers、目标预分配、all-rank gate 与 pool-internal `move_kv_cache`，但 healthy-path
shadow 不能复用其 request lifecycle。最小 adapter 从同一物理 allocator reserve private
pages、记录 destination generation、保持无 request/radix mapping；P0 恢复到 fresh request
pages后经 quarantined all-rank route permit发布，P1 才优化为 zero-copy ownership transfer。
这是一项 checkout-backed implementation mapping，不是已经接线或跑通的证据。

Router continuation 的 live-code mapping 同样已收窄。pinned SGLang 的 `Req.output_ids`
是 append-only；penalty orchestrator 通过 accepted output IDs 重建累计状态，grammar object
对 accepted token 调用 `accept_token`，deterministic sampler 用 per-request
`sampling_seed` 与 `forward_batch.positions`。因此 V1 不需要宣称序列化任意 GPU RNG：
Router 原子记录 accepted token、next absolute sampling position 与可重建 state digest，
target 从 token history 重新推进标准 penalty/grammar/stop state。custom processor 或非标准
position semantics 没有 serializer/replayer 时 fail closed。这也是 checkout-backed mapping，
不是 Router WAL 或 hard-kill continuation 已实现的证据。

对 live checkout 的接口核对还给出了一个重要边界。Mooncake 的
`ReplicateConfig.group_ids` 会让 members 共享 metadata routing、lease refresh 与
memory eviction，`PutStart/PutEnd`、object checksum 和 `BatchGetReplicaList` 可以复用；
但 `BatchPutEnd` 在 Master 中逐 key 调用 `PutEnd` 并返回逐 key结果，故不提供 Aegis
所需的整组发布点。SGLang `MooncakeStore.batch_set_v1/v2` 已能把真实 host KV buffers
交给 `batch_put_from`、并用 `batch_get_into` 读回；它现在仍按 raw component existence
判断 prefix，且 group semantics 需要显式 extra config。Aegis 的最小 runtime adapter
因此只需补 compatibility/security namespace、marker-last record 验证和 recovery-only
group lease/barrier，而不能把现有 HiCache hit 直接当成 recovery commit。

已有的 `benchmark/aegiskv_mooncake/results/66924-20260803-140423` 是一条可复核的
**数据面可行性**产物：Qwen3-8B、DP=2、TP=1 的两个 rank 都从 Mooncake 读回 960-token
prefix，记录了 150,994,944 B backup 与 283,115,520 B prefetch。它没有 source hard
kill、commit record、continuation/outbox 或请求续跑，因此不通过 G3，更不是容错性能
结果；它只说明最小纵切可以复用现有 actual-byte put/get，而无需先重写传输引擎。

## 5. 必须实验回答的内容

理论不能替代下列现实性与收益问题；但只需围绕主张设计最小充分实验集。

1. 真实 multi-agent workloads 中，positive hard deadlines 与 `p_protected<s<p_unprotected` crossing 是否足够常见。
2. 与 matched LUMEN-style load/size、TokenCake/PBKV-style agent priority、Continuum-style TTL、CacheFlow-style physical planner/longest-remaining-recompute order、SPT/EDF、MFS-inspired defer/promote、TENT-native priority 相比，Aegis 是否在 held-out traces 上稳定改变 decision；比较 CacheFlow 时必须把 planner 与 outer protection/deadline policy 两个因素分开。不同决策是否改善 workflow WCT/SLO 则由单独的 paired hard-fault runs 回答。
3. peer-HBM redundancy 的 no-fault HBM/traffic/TTFT/TPOT 成本与 failure-rate break-even。
4. Mooncake/SGLang 物理数据面上的 KV copy/load/adoption、hard process/GPU kill 与真实恢复时间。
5. `F_serve/F_recover` 隔离是否兑现，quantum terminal time 是否确实有有限 `B_g`；否则必须报告 `UNISOLATED`。
6. AIMD 在 load step 和 emergency pressure下的响应速度、振荡与正常 serving 影响。
7. conservative deadline-risk reduction 最终有多少转化成 realized workflow WCT/SLO benefit；理论上不能由 lower bound 自动推出。

可以复用四组数据完成这些问题：workload opportunity trace、policy/oracle replay、真实 hard-kill paired runs、no-fault/step-response runs。无需为每个 theorem 单独再做一组实验。

## 6. v2 正文重写结构

建议新正文按以下顺序展开，并控制在三个算法和四个组件之内：

1. **Overview and scope**：一句话问题、四组件图、E1 fail-stop 与同-layout scope。
2. **KV state and correctness**：content/lifecycle identity、sealed chunk、group commit、shadow adoption、output fence。
3. **Algorithm 1: Causal Slack Tracker**：只发布 `BOUNDED/UNKNOWN`；absolute deadline 与 invalidation。
4. **Algorithm 2: Slack-Deadline KV Manager**：action space、三段 value、无参数 robust risk、dominant-share admission/exchange、fixed-target dynamic EDF、DEFERRED fallback、离线 oracle。
5. **Algorithm 3: AIMD Residual-Resource Controller**：`F_serve/F_recover/background` 三层、HBM ledger、逐资源 token bucket、emergency reclaim；AIMD 只调 residual 利用率，不承担 safety proof。
6. **Implementation mapping**：Router/Conductor、Aegis Controller、Store Master、LLM Worker 的 API 与正常/故障时序；不在图或正文中把 substrate 品牌写成贡献。
7. **Guarantees and limitations**：T1--T5 各一段结论与 premise；`UNISOLATED`、跨 layout、相关多故障、Router failure 明确降级。

## 7. 写作守则

- 不再使用 Safe Frontier、manifest、complete backup DP、scenario optimizer 作为核心术语。
- 不把 DRAM/SSD 称为 Aegis backup；它们是已有 Store state。
- 不把 EDF、greedy、AIMD 本身写成 novelty；新意是 workflow deadline 对完整 KV action/data plane 的跨层控制。
- 不把 laxity-aware flow scheduling、deadline 字段、multi-path transport 或 transport self-healing 写成 novelty；MFS/TENT 已覆盖相邻机制。
- 不把 `BOUNDED` harm upper bound 的下降量写成 guaranteed WCT saving；它是保险式 deadline-risk score，真实收益用 paired faults 测。
- 不声称 EDF 优化整个两维 risk；其定理只覆盖 maximum lateness/tardiness 第一项。
- 不声称 enum 中存在 `BOUNDED` 就等于 positive hard-bound provider 已实现。
- 不声称 SGLSim metadata semantics 等于真实 tensor transfer。
- statistical Slack 已从 submission core 删除；G2 记为 N/A，不再作为等待补实验的分支。
- 不为理论命题堆砌实验；也不以理论替代 opportunity、cost 和 end-to-end outcome。
