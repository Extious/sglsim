# AegisKV 理论附录：T1–T5 的命题、反例与证明边界

> 状态：设计级 formal appendix；不等价于物理实现或 workload 证据  
> 对应主设计：[current_design.md](current_design.md)  
> 目标：把不需要性能实验的结论先证明清楚，并明确哪些前提必须由实现检查。

## 1. 模型与记号

在观测时刻 `t_0`，workflow runtime 暴露的 proof scope 可以是完整 residual DAG，也可以只是当前 branches 到下一个 mandatory AND-join 的 closed region。后者不要求知道 join 之后的未来：公共 suffix 同时平移所有 branch paths，在 Slack 差中抵消。节点是 recovery context supernode：同一 causal chain 上必须共同恢复的剩余工作先合并，frontier contexts 构成 antichain；closed region 中每个 node 都必须到达该 barrier。graph/work bounds 与 service minima 由同一个 fenced witness contract 覆盖，其绝对有效期至少到 scope 的 longest upper completion horizon。固定该 scope 的**原始** source/sink 集合；删除 `i` 时不能把它的 predecessor 或 successor 当成新的 source/sink，否则会把不完整半路径误当成 workflow completion path。对当前 frontier 上的 context `i`：

- `T_i`：从 `t_0` 起、**不含 failure-induced pause/rebuild 区间**时，原始 source-to-sink paths 中经过 `i` 的真实最长剩余 continuation time；
- `T_{-i}`：原始 source-to-sink paths 中不经过 `i` 的真实最长剩余路径时间；若不存在则取 0；
- `U_i`：`T_i` 的 hard upper bound；
- `L_{-i}`：`T_{-i}` 的 hard lower bound；
- `s_i=[T_{-i}-T_i]^+`：真实 structural Slack；
- `s_i^B=[L_{-i}-U_i]^+`：Aegis 发布的 hard Slack lower bound。

故障后，job `j` 被 Router 固定到 target `g(j)`：

- `O_j`：workflow eligibility origin；active job 为 failure cutoff，deferred job 为 dependency-ready event；
- `R_j`：failure fencing、target availability、recovery-state pin/read lease 和 dependency-ready 都满足后的 machine release time；
- `D_j=max(O_j,deadline_at_j)`：`BOUNDED` ticket 在 origin 上冻结 remaining Slack 得到的 absolute recovery due date；`UNKNOWN` 取 `D_j=O_j`；
- `Π_j(S)`：保护集合 `S`、固定 target/layout 和 resource envelope 下所有合法 physical recovery plans；
- `p_j(S)`：从 `Π_j(S)` 选中 plan 的 terminal makespan bound，即相对无故障 continuation 的 recovery service；
- `Σ_d(S)`：failure scenario `d` 中所有 jobs 在共享资源上的合法联合 schedules；pipelined mode 的 `{C_j(S)}` 必须来自同一个 `sigma in Σ_d(S)`；
- `C_j(S)`：target schedule 中的完成时刻；
- `\ell_j(S)=C_j(S)-D_j`：signed lateness。

`O_j`、`R_j` 与 `D_j` 语义不同。dependency-ready 可以成为 deferred job 的新 origin；target availability 只改变 `R_j`，不能把 deadline 延到 `R_j`。这里 `C_j` 是 branch 再次具备正确可继续状态的绝对时刻；detection、fencing、target wait、load/recompute、adoption、silent replay 与恢复 barrier 要么通过更晚的 `R_j`，要么通过 recovery service 进入 `C_j`。`U_i` 只界定 `C_j` 之后尚需的 continuation。若 target slowdown/catch-up 被计入 recovery service，就不能再计入 `U_i`；若留在 `U_i`，planner 就不能重复收费。这个切分必须由一个显式 resume boundary 固定。

本附录的 deterministic 结论只覆盖：fail-stop、固定 target、同 layout、commit 后 immutable 的 KV，以及明确写出的 resource envelope。Recovery Planner 的集合单调性允许 job 内部并行；EDF 的定理另外只覆盖单 target 串行 logical lane，串行 prefix DP 的定理另外只覆盖 edge-additive operation grammar。统计 Slack 不在 EuroSys submission core；跨 target assignment、未隔离 shared-resource interference 和 workload prevalence 也不在定理范围内。

逐节点 lower/upper duration witness 足以构造上述两个量：`L_{-i}` 是所有避开 `i` 的原始 source-to-sink paths 上 node lower bounds 之和的最大值；`U_i` 是所有经过 `i` 的原始 paths 上 node upper bounds 之和的最大值。若真实 node duration 落在各自区间内，则最长值的单调性分别给出 `T_{-i}>=L_{-i}` 与 `T_i<=U_i`。图不完整、任一 bound 缺失或 version 不一致时不满足前提，必须整图输出 `UNKNOWN`。

node interval 也必须有可审计来源。对同一 work unit，若 `W^-<=W<=W^+`、`0<mu^-<=mu<=mu^+`、固定开销 `f^-<=f<=f^+`，且 queue wait `0<=q<=Q^+`，则

\[
f^-+\frac{W^-}{\mu^+}
\le f+q+\frac{W}{\mu}
\le f^++Q^++\frac{W^+}{\mu^-}.
\]

所以这两端可分别作为 node duration lower/upper bound。若多个 nodes 共享资源 `r`，contract admission 必须联合验证其同时有效的 floors 与既有 guarantee 之和不超过 `C_effective,r`，并为共享 scheduler 给出 region-level 或经证明可组合的 queue bound；否则逐 node interval 即使各自有限也不能合成为 path witness。没有正的 reserved `mu^-` 或有限 `Q^+` 时右端不存在，不能用 mean/p95 填补；work 与 rate 单位不一致时该不等式无定义。

---

## 2. T1：因果可辨识性与 hard deadline

### 定理 1（无未来约束时，正 Slack lower bound 不可识别）

令 `H(t_0)` 为 Controller 在 `t_0` 前可见的全部因果历史。若 continuation 不受额外约束，则不存在仅依赖 `H(t_0)`、对所有 continuation 都正确且有时输出正值的 Slack lower-bound estimator。

**证明。** 固定任意历史 `h`。构造两个在 `t_0` 前都等于 `h` 的 continuation：

1. `c_0` 中 sibling 立即结束，使 branch `i` 的真实 Slack 为 0；
2. `c_\Delta` 保持 branch `i` 的 continuation 不变，但令 sibling 的剩余时间至少为 `T_i+\Delta`，其中 `\Delta>0`，使同一 branch 的 Slack 至少为 `\Delta`。

任何 `f(H(t_0))` 在二者上输出相同值。若输出 `x>0`，它在 `c_0` 上不是 lower bound；若总输出 0，则没有正 utility。对 randomized estimator，任何正值具有非零概率都会破坏 almost-sure lower-bound claim。故分布无关的安全输出只能是 0。证毕。

这个定理否定的是“从相同 causal prefix 必然推出正 Slack”，不是否定统计预测。统计模型可以依赖分布假设，但必须标为 probabilistic，并单独检查 selection-aware coverage。

### 引理（mandatory-join 之后的公共 suffix 可以省略）

设当前 closed region 的所有 branch 必须到达 AND-join `J`，`J` 之后到 workflow terminal 的剩余执行对这些 branches 是同一个公共 suffix，且其 work/queue behavior 不因哪个 branch 最后到达而改变。令 branch `i` 到 `J` 的时间为 `P_i`，其他 branches 中最慢者到 `J` 的时间为 `P_{-i}`，公共 suffix 时间为 `Q`。则

\[
[(P_{-i}+Q)-(P_i+Q)]^+=[P_{-i}-P_i]^+.
\]

因此只声明到 `J` 的 closed region 足以计算 structural Slack；不需要预测 `J` 之后的 future turns。若 join 是 OR/quorum、branch 可以 cancel/spawn，或 downstream batching/route 取决于最后到达者，`Q` 不再是公共项。此类动态 region 只有在改变生效前先由 Controller 完成 version fence 和 ticket downgrade 时才能发布 hard witness；best-effort asynchronous invalidation 不满足引理前提。证毕。

### 引理 1（hard-bound Slack 安全性）

若 `T_i<=U_i` 且 `T_{-i}>=L_{-i}`，则 `s_i>=s_i^B`。

**证明。**

\[
T_{-i}-T_i\ge L_{-i}-U_i.
\]

函数 `[x]^+=max(x,0)` 单调不减，因此

\[
[T_{-i}-T_i]^+\ge[L_{-i}-U_i]^+.
\]

证毕。

### 引理 2（正 hard-Slack ticket 的 barrier expiry）

假设 `L_{-i}>=U_i`，并且 fenced witness contract 有效期间：

- sibling path 不可能早于 `t_0+L_{-i}` 完成；
- recovery pause 结束后，branch `i` 至多再需 `U_i` 完成；
- contract 的绝对有效期覆盖 `t_0` 到声明 scope 的 upper completion horizon；
- 任何会削弱这两个 witness 的 graph、route、queue 或 service change，要么在 Controller 原子撤销 ticket、把故障中相关 due date 收紧到 `min(old_deadline,event_time)` 并 ACK fence 后生效，要么 authority 的 version bump 在同一线性化点让所有 consumer 将旧 ticket fail closed 为 zero Slack；异步重算不能保留旧数值。

前两项必须按 failure-conditioned counterfactual 量化：sibling lower bound 要在故障后可能获得的最快合法服务下仍成立；target-path upper bound 要覆盖从显式 resume boundary 到下一同步点的最慢受约束 continuation。resume boundary 之前的 reroute、恢复竞争、load/recompute、adoption 和 replay 由 `C_i` 计时，不能再塞进 `U_i`。健康 trace 的均值或只对原 placement 成立的 bound 不满足该前提。

则

\[
D_i^B=t_0+L_{-i}-U_i=t_0+s_i^B
\]

是正 hard-Slack ticket 的保守 absolute expiry。

**证明。** 即使悲观地假设 branch 在故障前没有取得任何进展，只要 pause 在 `D_i^B` 前结束，它最迟在

\[
D_i^B+U_i=t_0+L_{-i}
\]

完成，不晚于 sibling 的最早完成时刻。故障前已完成的 branch work 只会缩短剩余时间，不能使该 expiry 不安全。检测、fencing、控制消息和 target waiting 都发生在同一绝对时间线上，所以在任一时刻 `t<=D_i^B` 最多只能消费剩余量 `D_i^B-t`，不能重新加回原始 `s_i^B`。证毕。

若 `L_{-i}<U_i`，本引理的前提不成立。此时定义仍给出 `s_i^B=0,D_i^B=t_0`，但它只表示没有可证明的正隐藏时间，不能声称“零暂停即可早于 sibling”；尤其不能把带正部的 `t_0+[L_{-i}-U_i]^+` 替换成更早的 `t_0+L_{-i}-U_i`。

### 推论（failure cutoff 只冻结 remaining Slack）

令 active job 的 failure cutoff 为 `F>=t_0`。定义

\[
s_i^{rem}(F)=[D_i^B-F]^+,
\qquad
D_i^F=F+s_i^{rem}(F)=\max(F,D_i^B).
\]

若 `F<=D_i^B`，引理 2 直接证明在 cutoff 后至多还有 `D_i^B-F` 的可隐藏 pause；若 `F>D_i^B`，零永远是安全的 Slack lower bound。因此 `s_i^{rem}` 是 cutoff 时的 remaining-Slack lower bound。这里把 due date clamp 到 `F` 不是给旧 ticket 充值：过期 ticket 得到的正余量仍为 0；它只避免把故障前的正常年龄 `F-D_i^B` 误记成 recovery lateness。cutoff 后的 detection、fencing、target wait 和 recovery work继续让 `C_i` 变晚，并且 `D_i^F` 不得由 post-failure prediction 延后。

### 引理 2a（cutoff-normalized absolute lateness 上界 workflow-visible harm）

令 `P_i` 为故障相对无故障路径在同一 synchronization boundary 引入的全部额外 shift，真实 structural Slack 为 `s_i`。Algorithm 2 的 terminal completion `C_i` 必须覆盖 detection、fencing、target wait、load/recompute、adoption、silent replay、barrier，以及 recovery target 相对 source 到该 boundary 的全部 catch-up，使

\[
P_i\le C_i-F.
\]

若某段 failure-only slowdown 留在 `C_i` 之后，则这个前提不成立，不能同时把它留给 `U_i` 又声称 `C_i-D_i^F` 已完整收费。令 `s_i(F)` 为 cutoff 时真实 remaining structural Slack；上面的推论给出 `s_i(F)>=s_i^{rem}(F)`。path-shift 模型于是给出

\[
\Delta_i
\le[P_i-s_i(F)]^+
\le[P_i-s_i^{rem}(F)]^+
\le[C_i-F-s_i^{rem}(F)]^+
=[C_i-D_i^F]^+.
\]

所以 Algorithm 2 使用 absolute completion `C_i` 与 cutoff-normalized due date `D_i^F` 的 tardiness，不需要把 detection、target wait 或 queue time再构造成一份相对 Slack；它们只要让 `C_i` 变晚就已被完整收费。当 `F<=D_i^B` 时，引理 2 还给出直接 barrier 解释；当 ticket 为零或已过期时，结论退化为平凡但安全的 `Delta_i<=[C_i-F]^+`。对同一 workflow 的 recovery frontier antichain，在引理 4 的 path-shift/no-extra-interference 前提下再取 `max_i` 即得 workflow harm 上界。这个不等式是解析结论，不需要故障实验；实现只需证明 `C_i` 的 terminal ACK 覆盖定义同步边界前的全部 failure-only work。

注意，仅在 event log 中递增 version 不足以满足最后一项：若改变已经生效而撤销消息仍在网络中，旧 deadline 在该窗口内可能把真正紧急的恢复排后。这是协议顺序条件，不是可以用低 stale-rate 实验近似掉的统计误差。

### 必要反例：receive time 不是 event time

dependency 在远端 `q=0` ready，Controller 在 `q=3` 才收到消息。若用 receive time 作为 UNKNOWN job 的 deadline，就把 3 秒 control delay 免费隐藏。proof mode 必须使用带误差上界的同步 timestamp，或以 `receive_now-delay_upper_bound` 取最早可能时刻；两者都没有时只能 fail closed。

### 必要反例：workflow SLO 本身不是 recovery Slack

令两个 executions 在 `t_0` 前具有相同 causal history 和同一个绝对 workflow
SLO `T_w`。在第一个 continuation 中，recovery completion 后只剩 1 ms mandatory
work；在第二个中还剩 50 ms。若 Controller 只看 `T_w`，就会在二者上发布相同
recovery due date；例如 `T_w-10 ms` 在第一个 execution 可满足 SLO，在第二个却
必然超时。更一般地，没有 post-recovery remaining work 的有限 hard upper bound
`U_post`，SLO 不能推出一个保证完成的 latest recovery time；若该 bound 存在，
`D_i^{SLO}=T_w-U_post` 才是 SLO-feasibility deadline。

它与本节的 structural-hiding deadline 语义不同：`D_i^B` 回答故障暂停是否仍被
sibling path 隐藏，`D_i^{SLO}` 回答整个 workflow 是否还能在外部 SLO 前完成。
若部署要求同时满足二者，安全 due date 是两者中更早者，但一个裸 `optional SLO`
不能把 `UNKNOWN` ticket 升级为 `BOUNDED`。Aegis core 因而只用 structural ticket
排 recovery，并在 outcome 层用 SLO headroom 计算 tardiness；未来若引入 SLO due
source，必须单独携带上述 remaining-work contract 和 provenance。这个信息缺失反例
是理论边界，不需要 workload 实验；真实 contract 覆盖率才需要测量。

### T1 的证据责任

- 上述不可辨识性、closed-join、hard-bound 与 absolute-deadline 结论：理论分析即可；
- mandatory-join/closed-region 语义、witness 有效期以及 invalidate-before-effect fence 是否由应用和 service envelope 真正提供：接口/state-machine validation；
- positive hard deadlines 在真实 workflow 中是否常见：G1 trace measurement，不能由定理回答。

---

## 3. T2：Slack deadline risk、action value 与非等价性

### 引理 3（单 job 的精确三段收益）

令一个 recovery action 把额外 service 从 `p_0` 降到 `p_1<=p_0`，真实 Slack 为 `s`。其 workflow-visible harm 为 `H(p,s)=[p-s]^+`，action value 为

\[
\Delta H=[p_0-s]^+-[p_1-s]^+.
\]

逐段展开得到：

\[
\Delta H=
\begin{cases}
0, & p_0\le s,\\
p_0-s, & p_1<s<p_0,\\
p_0-p_1, & s\le p_1.
\end{cases}
\]

这是一条恒等式，不需要 trace 或 GPU 实验。

### 推论（hard Slack lower bound 不是 realized-benefit certificate）

若 online 只知道 `s^B<=s`，定义 deadline-risk upper envelope

\[
H^B(p)=[p-s^B]^+.
\]

由正部函数单调性，`H^B(p)>=H(p,s)`；而且对所有允许的真实 Slack `s>=s^B`，有

\[
H^B(p)=\sup_{s\ge s^B}H(p,s).
\]

因此使用 `s^B` 计算的

\[
\Delta H^B=H^B(p_0)-H^B(p_1)
\]

是两个 worst-case harm upper envelopes 的差。它是合理的 insurance/deadline-risk objective，却不是实际收益 `\Delta H(s)` 的 lower bound。事实上，引理 3 的三段函数对 `s` 单调不增，所以 `s^B<=s` 推出

\[
\Delta H^B\ge \Delta H(s).
\]

最小反例为 `s^B=0,s=10,p_0=5,p_1=0`：`\Delta H^B=5`，但两个真实 harm 都为 0，故 realized benefit 为 0。若 `p_0,p_1` 也只是分别计算的 service upper bounds，它们的差同样不是 realized service saving 的 lower bound。

所以 hard-bound Aegis 能证明的是：meeting the published deadline 足以隐藏 pause，以及某 action 降低了这个 deadline 下的 worst-case risk；它不能由此证明该 action 在实际 continuation 上节省了多少 WCT。前者只需理论分析，后者必须由 matched paired failure outcomes 测量。

### 反例 1（与 size/load score 不等价）

考虑两个 bytes、source load、target load 和 copy cost 完全相同、且表中 `s,p_0,p_1` 取精确值的 action：

| action | `s` | `p_0 -> p_1` | `ΔH` |
|---|---:|---:|---:|
| A | 2.0 s | 1.5 -> 0.2 s | 0 |
| B | 0.5 s | 2.0 -> 0.2 s | 1.5 s |

任何只看 size/load 的 score 对 A、B 打平；Slack-Deadline 必须选 B。这只证明决策函数不等价，不证明真实 workload 经常产生该 case。

### 反例 2（与 binary criticality 不等价）

- critical job：`s=0, 0.10->0.05 s`，收益 `0.05 s`；
- noncritical crossing job：`s=0.50 s, 2.0->0.2 s`，收益 `1.50 s`。

binary criticality 会优先第一个，Slack action value 优先第二个。因此“是否在 critical path”不足以表达 recovery actuator 的大小。

### 引理 4（frontier `max` 聚合）

令 workflow 无故障 makespan 为 `M`，当前 recovery frontier 是 DAG antichain；job `j` 所在路径原完成时刻为 `P_j<=M`，recovery 给该路径增加 `x_j`。在路径只被平移、没有额外 shared-resource slowdown 时：

\[
\Delta WCT
=\max\left(0,\max_j(P_j+x_j)-M\right)
=\max_j[x_j-(M-P_j)]^+.
\]

因此 sibling jobs 应按 `max` 聚合，而不是求和。若使用 lower bound `s_j^B<=M-P_j`，得到的 `max_j[x_j-s_j^B]^+` 是 deterministic conservative upper bound。

**边界反例。** 两个 lost states 在同一 causal chain 上，各引入 1 秒 pause。真实路径增加 2 秒，逐 job `max(1,1)=1` 会低估。因此 ancestor/descendant states 必须 coalesce 成一个 recovery job，或让后继保持 `DEFERRED`；不能把该引理推广到任意 job set。

### 反例 2a（exact Slack/service 不足以让旧 absolute ticket 变成 realized WCT）

在 `t_0=0`，并行路径 A、B 的真实剩余时间分别为 10、20 秒，所以 A 的 exact structural Slack 为 10，ticket expiry `D_A^B=10`。两条路径正常运行 9 秒后 A 所在 worker 失败；cutoff-normalized due date 仍为 `D_A^F=max(9,10)=10`。A 的 failure-induced pause 为 2 秒，故 recovery terminal absolute time `C_A=11`。之后 A 还有 1 秒正常 work，在时刻 12 完成；B 仍在时刻 20 完成。因此 workflow 无故障/故障 makespan 都为 20，真实新增 WCT 为 0，但 conservative risk 给出 `[C_A-D_A^F]^+=1`。

引理 4 的等式本身没有错：在 failure cutoff `t_*=9` 重新取 common origin，A 的真实 Slack 仍为 10、纯 path shift `x_A=2`，所以 `[x_A-s_A]^+=0`。cutoff clamp 只删除已经为负的 ticket 余量，不能凭旧 hard lower bound推断故障时的 exact Slack；这里它安全地保留 1 秒而不可能恢复未知的 10 秒。只有 `D_j=t_*+s_j,C_j=t_*+x_j` 的 same-origin exact instance 可以把 `H_w` 写成 realized WCT；部署的早期 hard ticket 与 conservative terminal bound 只给 upper-risk score。该反例无需实验；progress refresh 频率与 bound tightness 才需要测量。

### 引理 5（未 commit、未 pin/lease action 的恢复收益为 0）

正常期 protection-visible set `V` 只包含 complete group commit；reservation、已发命令或 partial shard ACK 都不能提前产生 realized recovery value。实际故障时 recovery-visible set `V_R` 还只包含本次 route 上成功 peer-pin 或 Store read-lease 的 groups。只要 action `a` 尚未进入对应集合，恢复路径就不能读取它，因此把 in-flight metadata 放进当前真实状态不会改变 `p_j`，realized `Gain(a|S)=0`。特别地，`COMMITTED` 但 `FreezeAndPin`/read lease 失败的 action 对该次故障的 realized recovery value 仍为 0。

这里还要区分 fail-now presence 与 horizon guarantee：当前确实存在的 Store replica 可进入时刻 `t` 的 fail-now snapshot；只有 quota-backed tier lease 才能证明它在未来 `tau` 仍位于不低于 `min_medium` 的层。soft cache priority 既不扩展 deterministic horizon set，也不能给 Retain/Accelerate 提前记收益。

启动 copy 前的 policy score 是另一件事：`S\oplus a` 只表示 action 在 completion horizon `tau_a` 已完整提交，不能单独定义那一刻的 workflow/job state。`S` 中一个 existing action 若在 `tau_a` 前过期，其 effect 不能进入 horizon risk；但在 release/reclaim terminal ACK 前，它的 resource demand 仍被占用。这避免一边把 expired copy 当作保护，一边把尚未释放的 bytes 当作免费容量，也不给未 commit bytes 任何 realized value。

### 反例 3（current snapshot 不能识别 post-commit action value）

在 `t` 给定完全相同的 revealed history、job、deadline、prefix 与一个完成上界 `tau_a>t`。continuation A 中 job 至少活到 `tau_a`，action commit 后把 recovery service 从 12 秒降到 2 秒；continuation B 中 job 在 `tau_a` 前正常完成并释放，故障时不再产生该 recovery job。两者在 `t` 的全部输入和 copy completion bound 相同，但 post-commit value 分别可能为正和 0。也可以让两者都存活、但在 `tau_a` 前生成不同长度的 tail，使 crossing 区间不同。

因此，仅有当前 `H(t)`、`S\oplus a` 与 copy-time bound 不能精确识别 `R(X_{\tau_a},S)-R(X_{\tau_a},S\oplus a)`；还需要同时约束 job liveness、prefix growth 与 service 的 horizon contract。job 至少活到 `\tau_a` 的下界可由 mandatory remaining work 和 fastest legal service 得到，但它仍不约束 tail/service 演化。若完整 contract 存在，Controller 对所有允许 horizon states 取最小 risk reduction；没有时只能使用明确标记的 myopic snapshot score，并在 sub-prefix/terminal commit 时按真实状态重算。该不可辨识性由上述两个 continuation 证明，不需要 workload 实验。

更精确地，若 contract 给出允许状态集合 `\mathcal X_a`，每个 horizon state 都以 `tau_a` 为共同 fail-now origin，并对 ticket expiry `E_j` 使用 `D_j(tau_a)=max(tau_a,E_j)`；也就是只把 positive remainder 消耗到零。certified score 定义为

\[
G^{cert}(a\mid S)=\min_{X\in\mathcal X_a}^{lex}
\left(R_X(S)-R_X(S\oplus a)\right),
\]

其中 pair subtraction 按分量进行、最后按 lexicographic order 取最小；`S` 的旧 recovery plan 必须在 `S\oplus a` 中保留为 null option。这样每个允许状态的 committed-state addition 都不会让 risk 在 lexicographic order 下恶化，故 `G^{cert}\succeq_{lex}(0,0)`；这不保证两个坐标都非负，因为第一维严格下降时第二维可能上升。若 action 同时撤销 victims，或 resource envelope/target 随之改变，就不能套用这个 inclusion；必须使用正文的完整 exchange 与 phase gate。

### 反例 3a（soft Store priority 不是 deterministic protection）

在时刻 `t`，同一个完整 Store object 位于 DRAM，Master 对同一 soft-priority hint 返回成功。构造两个都符合该 hint 语义的 continuation：A 在 `tau>t` 前因普通 cache pressure 把 object 降到 SSD，B 则让它继续留在 DRAM。若故障发生在 `tau`，两者的 recovery service 与 deadline risk 不同，但 Controller 在 `t` 看到的 object、hint 和 ACK 完全相同。因此 soft priority 不能把 `S\oplus a` 定义成一个确定的 failure-time medium，也不能进入 hard service bound。

要获得 deterministic horizon state，Master 必须可以拒绝请求，并在 quota reservation、必要 placement 与 ACK 后承诺 `(object/group,Master-selected physical replica set,protected source domain,min_medium,expiry)`；Controller 只给 failure-domain/placement constraints，不能凭缓存位置替 Master 指定或伪造 set。placement 必须位于 source failure domain 之外，`expiry` 覆盖评价 horizon。容量按 Master 选定并将 pin/reserve 的 marker/member replicas **逐层 actual physical bytes** 计费；只有 ACK 返回同一 replica-set digest 的引用可共享，内容相同但 placement 不同不能去重。SSD→DRAM promotion 在 PENDING 时已为 staged destination 预留 DRAM，源 SSD 的 bounded read pin/transfer 另计 demand；PROCESSING replica 仍为零 recovery value。否则该状态只能在真实故障时以当下 lookup + read lease 加入 `V_R`，或进入明确标记的 heuristic model。这个信息论反例无需性能实验；实现验证只检查 quota、placement ACK、expiry 和 failover 后是否兑现承诺。

### 引理 5b（tier-lease expiry 不需要跨机器时钟比较）

Controller 在自己的 monotonic clock 上于 `t_s` 发请求、于 `t_r` 收到 TTL 为 `T` 的 ACTIVE ACK。设 Master 的 grant 线性化点 `g` 在请求发送之后，ACK 只在 placement、quota 与 lease record durable 之后返回；Master 从 `g` 起至少保留 `T`。若 Controller 对该区间的 clock-rate/timestamp guard 为 `epsilon_clk`，则

\[
e_C=t_s+T-\epsilon_{clk}
   =t_r+T-(t_r-t_s)-\epsilon_{clk}
\]

是 Controller 时间轴上的保守 expiry lower bound。若 `e_C<=t_r`，ACK 没有可用的未来 horizon；否则只有 `tau<e_C` 的 action state 可进入 deterministic `S`。

**证明。** 由因果顺序，真实时间上的 `g` 不早于 send event，所以 `g+T` 不早于 send event 后 `T`；Controller 只用本地 elapsed interval 表示这个下界，再减去已建立的 clock guard。网络请求与 ACK 的往返已经体现在 `t_r-t_s` 中，无需读取或比较 Master absolute timestamp。若 ACK 到达时该 interval 已耗尽，就不存在可发布的正剩余保证。证毕。

renewal 只有在新 ACK 自身满足相同条件时才有效。若 Master 明确把 expiry 更新为 `max(old_expiry,g+T)`，Controller 可取旧、新两个安全点的最大值；若是 replace 语义，只能使用新安全点。这个结论是因果/时钟区间推导，不需要性能实验。

### 推论（Master failover 不能缩短已 ACK tier lease）

若 (1) ACTIVE record、actual-byte quota 与满足 `min_medium` 的 placement 在 ACK 前 durable；(2) delete/demotion 只能发生在 durable `EXPIRED/RELEASED` 之后；(3) 新 leader 以更高 authority epoch fence 旧 leader；并且 (4) 对每个仍为 ACTIVE 的记录，从 takeover 起重新保留至少一个原 TTL，或恢复一个可证明不更早的 cluster-time expiry，则 leader failover 不会使旧 ACK 的有效区间提前结束。

**证明。** takeover event 晚于原 grant，因此 `takeover+T >= g+T`。把 ACTIVE lease 从 takeover 起重启完整 TTL 只可能延长保护；durable terminal ordering 保证已经 demote 的对象不会仍以 ACTIVE 被恢复，epoch fence 保证旧 leader不能并发破坏新 leader 的承诺。PENDING 从未产生 protection ACK，故可在 failover 时 fail closed。证毕。

该安全做法可能在 failover 后多保留至多一个 TTL 的容量，这是 liveness/efficiency 成本而不是 safety 漏洞。若实现的 snapshot/oplog 丢弃 lease 或 quota，或 failover 后直接把对象当作普通 cache，则它不满足本推论，不能给 theorem-mode score 提供 tier guarantee。

### 引理 5a（Recovery Planner 的 null option 保证 service 单调性）

对固定 job、target/layout 和 resource envelope，令 `Π_j(S)` 是 failure-visible state `S` 下所有合法 physical recovery plans 的集合，`M^U(π)` 是 plan `π` 的 terminal makespan upper bound，并定义

\[
p_j^{*,U}(S)=b_j^U+\min_{\pi\in\Pi_j(S)}M^U(\pi).
\]

若 `S ⊆ S'` 且新增 state 不撤销旧 state 的 commit、pin/lease 或资源契约，则任何 `S` 下的 plan 在 `S'` 下仍合法，即 `Π_j(S) ⊆ Π_j(S')`，因此 `p_j^{*,U}(S')<=p_j^{*,U}(S)`。一个不求全局最优的实现 planner 也可通过把上一次可执行 plan 作为 null option，并只在新 plan bound 更小时替换它，保证其**返回 bound**单调不增。

这是集合包含关系，不需要性能实验。它不说明 planner 找到全局最优，也不说明两个 upper bounds 之差是 realized saving；实际 plan、resource envelope 和 terminal ACK 是否兑现仍需实现验证与 calibration。

### 命题（串行 multi-tier prefix recovery 是 DAG shortest path）

对一个 cutoff 含 `N` 个 KV groups 的 job，令顶点 `k` 表示 target 已得到 `[0,k)` 的正确连续 KV。每条有向边 `e=(k,m)` 满足 `m>k`，并表示一种已经通过 commit、pin/read-lease、layout 与 failure-domain 检查的原子操作：peer adoption/D2D、Store DRAM/SSD read，或对已知 token 的 silent recompute。边权 `c_e^U>=0` 是该操作在 recovery envelope 下的 service upper bound；最终 resume/barrier upper bound 为 `b^U`。则

\[
p^U(S)=b^U+\min_{P:0\leadsto N}\sum_{e\in P}c_e^U
\]

可由 token-boundary 拓扑序在 `O(|E|)` 时间求出。若 edge setup 依赖前一 medium，把顶点扩为 `(k,medium)` 后结论不变。

**证明。** 每条 edge 的前置状态是正确 prefix `[0,k)`，后置条件是正确 prefix `[0,m)`；所以沿 path 归纳，每个 source-to-terminal path 都是连续且合法的串行恢复 plan。反过来，在声明的 edge-additive atomic-operation grammar 内，任一合法 serial plan 的 successive boundaries 构成这种 path。边只向更大 boundary 前进，图为 DAG，标准 recurrence `P[m]=min_{(k,m)}(P[k]+c_{k,m}^U)` 给出最小 path weight。选中 path 中每个操作的实际 service 不超过其 edge upper bound，因此该 serial plan 的实际 service 不超过 `p^U`。增加一条仍有效的 edge只扩大可选 path 集，不会增大最小值。证毕。

这里有两个不可混淆的 lifetime：normal-time protection lease 决定 object 在 failure snapshot 时能否进入 `V_R`；成功的 recovery pin/read lease 随后接管 ownership。若 read lease 不是 lifecycle-held，而是有限窗口，则必须在 edge dispatch 时 renew/revalidate，使窗口至少覆盖 `c_e^U`；失败就删除 edge并重跑 DP。EDF queue wait 不能继续消耗已经被 recovery pin 接管的旧 protection TTL。Controller failover 只能把完整 leased key set rebind 到新 issuer，Store authority failover 必须恢复 refcounts或保守延长全部已 ACK keys；subset reconstruction 或丢 ref 后继续 load 都关闭该 edge。该命题不证明物理 edge 已 commit 或其 service envelope 可兑现；这些是 state-machine 与 targeted calibration 责任。它也说明 serial reference 中 portfolio action 的效果必须来自对整个 state 重跑 DP，不能把不同 tier 的独立 scalar savings 直接求和或取最大。

### 反例 4（串行 shortest path 不能表达一般并行 restoration）

一个 job 的前半 prefix 可用 GPU 在 5 秒内重算，后半 prefix 可同时用 I/O 在 5 秒内从一个已提交 Store object 读取；两项在不同资源上无依赖，只需在末尾 join。合法并行 plan 的 makespan 是 `max(5,5)=5` 秒。若把两项表示成从 boundary 0 到中点、再到终点的 additive edges，任何 path weight 都是 `5+5=10` 秒；反向 edge 顺序也一样。故普通 token-boundary shortest path 只对串行 operation grammar 精确，不能声称覆盖 token/layer/GPU overlap、cross-stage pipeline 或 batch-level contention。

这个反例本身已经否定一般等价性，不需要 GPU 实验。实验只需回答选用的并行 planner 在真实硬件上是否达到声明的 bound，以及 Aegis 的 deadline/protection policy 在固定该 planner 后是否有收益。

### 反例 4b（pipelined mode 不能拼接独立 per-job makespan）

有 jobs `A,B`，每个单独恢复都包含 5 秒 compute 与 5 秒 I/O，二者在 job 内可并行，所以独立最优 makespan 均为 5 秒。两个 jobs 却共享同一个 capacity-1 compute server 和同一个 capacity-1 I/O server；在区间 `[0,5]` 内每个 server 最多提供 5 秒服务，不可能同时完成两份各 5 秒的 demand。因此 completion vector `(5,5)` 不可行，即使两个独立 `Π_A,Π_B` 都给出 5。

所以 pipelined action value 必须在联合 schedule 集 `Σ_d(S)` 上求一个共同 completion vector，不能把 `min_{pi_A}M_A` 与 `min_{pi_B}M_B` 当成同时可实现。若 `S ⊆ S'` 且旧 schedule 仍合法，则 `Σ_d(S) ⊆ Σ_d(S')`；精确 lexicographic risk optimum 单调不增，heuristic 也可保留旧 schedule 作为 null option。这个 capacity 反例与集合包含证明不需要性能实验；实现只需验证 operation demand、共享资源约束和返回 schedule 的可执行性，经验评测再量化 heuristic gap。

### 反例 5（prefix dependency 破坏一般 submodularity）

一个 job 只有在 chunk 1、2 都可连续恢复时才避免 10 秒 recompute。令 `a_1`、`a_2` 分别保护两个 chunk：

\[
\operatorname{Gain}(a_2\mid\varnothing)=0,
\qquad
\operatorname{Gain}(a_2\mid\{a_1\})=10.
\]

边际收益随集合增大反而增加，违反 submodular diminishing returns。因此 multi-resource lazy greedy 没有通用 `(1-1/e)` 保证。

这个反例还给出一个实现要求：若 online greedy 只生成 `next-Extend` 并拒绝非正 gain，那么它会拒绝 `a_1`，此后永远看不到 `a_2`，即发生 **prerequisite starvation**。Aegis 因而把从当前连续边界到一个候选 terminal 边界的全部缺失 chunks 组成 dependency-closed bundle `b`，一次计算总 resource demand、closure completion upper bound 和 terminal effect。只要 `Gain(b|S)>0`，即使某个内部 chunk 单独 gain 为 0，整个 bundle 仍可被接纳。若执行途中更短 prefix 已真实 commit，它作为新 `S` 重算；未 commit bytes 不获得恢复价值。

bundle 化并不恢复一般 submodularity：两个 candidate bundles 可以重叠，共享 action 可同时改变多个 jobs，EDF completion/risk 又含 `max` 与 queue interaction。因此它只修复上述可达性 bug，不产生 approximation ratio；small exact/MILP oracle 仍需报告 regret。

### 在线 selection 的有界复杂度

online Controller 不求“最小 risk-loss cover”。对每个 candidate bundle，它按正文定义逐轮选择 `Loss/cover` 最小的 dependency-closed victim group，并在每次移除后重算 deficit、loss 和 cover；得到一个可行 victim set 后，才重算完整 exchange 的 terminal risk。令 `A` 为 endpoint cap 后的 changed candidates，`V` 为可释放 victim groups，`K` 为一个 protection window 允许的最多 admission/exchange steps，并令

\[
T_{eval}(a,S)=\sum_{j\in J(a)}T_{plan}(j,S)+\sum_{d\in D(a)}O(n_d\log n_d).
\]

一个 cover 至多进行 `V` 轮、每轮至多扫描 `V` 个 groups；每个 window 的保守上界因而是 `O(K A V^2 T_eval)`，空间为 selected-set/planner cache 所需的 `O(A+V+\sum_d n_d+\sum_j |E_j|)`，不含具体并行 planner 的私有 workspace。serial prefix planner 有 `T_plan=O(|E_j|)`；一般并行 planner 必须提供有限 timeout contract。candidate crossing/geometric endpoints 与 `K` 均受配置上限约束，wall-clock decision deadline 还可提前 fail safe。这个 bound 只证明算法不会隐式调用指数精确求解；它不提供 approximation ratio，也不排除 `T_eval` 本身因所选 general planner 而较大。

这里的 terminal set difference 不是物理原子交换。跨 owner 的 PREPARE 只在各自局部账本原子预留并以 TTL 回收；partial prepare 没有 protection value。若 candidate 不能在 victims 仍保留时完成，执行计划必须列出每个 terminal reclaim、candidate commit 和 abort outcome。对 phase `q` 的 committed set `S_q`，core 要求 `\widetilde R_t(S_q;t_q)\preceq_{lex}\widetilde R_t(S;t_q)`，并要求最终 `S-V+a` 严格改善；两侧都以 `t_q` 为同一 fail-now origin并使用 `max(t_q,E_j)`，所以比较不会由请求年龄伪造。否则拒绝这个 terminal proposal。这个 phase gate 至多增加 `O(V)` 次相同 evaluation，不改变 `O(K A V^2 T_eval)` 的保守阶数。它关闭了一个重要反例：若直接把容量 1 中 service 从 8 降到 1 的新 shadow 与旧 shadow写成原子交换，实际先删旧 shadow 后 copy 期间 service 会回到 10；故障落在该区间时，静态 `R(S)-R(S-V+a)` 高估了保护。没有额外 headroom 或零边际-loss victim 时，core 正确地拒绝此 swap。该反例和状态枚举不需要性能实验。

### 定理 1（精确 protection selection 是 NP-hard）

对每个 canonical、只枚举一次的单 source-domain failure scenario `d`，令 `R_d=(R_d^{max},R_d^{sum})`；在 operator 已固定 domain set、risk pool 与预算后，正常期的 weight-free minimax objective 为

\[
R(S)=\left(\max_d R_d^{max}(S),\ \sum_d R_d^{sum}(S)\right).
\]

即使 action 相互独立、只有一个标量容量约束，精确最小化这个 lexicographic objective 也包含 0--1 knapsack。

**归约。** 给定 knapsack items `(w_i,v_i)` 与容量 `W`，构造一个不可保护的 sentinel scenario `d_0`：它只有一个 deadline 为 0、service 为 `M>max_i v_i` 的 job，所以对任意 action set，第一目标至少且恰为 `M`。对每个 item `i`，另建一个独立 scenario `d_i`，其中唯一 job 的 deadline 为 0、未保护 service 为 `v_i`；action `a_i` 消耗唯一资源 `w_i` 并把该 service 降为 0。各 scenario 可位于不同 target，因而互不排队。

对任意可行集合 `S`，第一目标恒为 `M`，第二目标为

\[
M+\sum_{i:a_i\notin S}v_i.
\]

所以在 `\sum_{a_i\in S}w_i<=W` 下最小化第二目标，恰好等价于最大化 `\sum_{a_i\in S}v_i` 的 0--1 knapsack。故精确 protection selection 是 NP-hard。证毕。

该定理解释为何生产 Controller 不应在每次 graph/progress event 上求全局精确解；它不证明当前 greedy 在现实 workload 上接近最优。前者是纯理论结论，不需要实验；后者仍需用小实例 exact/MILP oracle 报告 gap。结合反例 5，当前 greedy 也不能借用 submodular maximization 的标准 approximation guarantee。

### 反例 6（weight-free minimax 不是 expected loss 或 fairness theorem）

有两个 failure domains `d_1,d_2`，故障时 harm 都为 10，同成本 action 分别把对应 harm 降到 0。若预算只够一个 action，Aegis 的 minimax 第一维在两者间对称，最终由第二维/稳定 tie-break 选择。现在令真实 hazard probabilities 为 `Pr[d_1]=0.999,Pr[d_2]=0.001`。expected-loss optimizer 必须保护 `d_1`，但一个没有 hazard 输入且在交换 domain label 后保持不变的 policy 无法从当前 risk state 推出这个选择。故 robust objective 不能声称 expected-loss optimality。

更一般地，lexicographic pair 会让任意正的 maximum-tardiness reduction 优先于任意大的 total-overrun reduction；这是 tail-SLO insurance 的明确定义，不是 tenant utility/fairness 定理。tenant/risk-class isolation 必须来自总和不超过物理 cap 的硬 quota pools，不能从 `rho` 或 lexicographic score 推导 DRF/strategy-proofness。这个 claim-scope 由定义和反例关闭；具体部署是否应选择该 insurance profile 才由 failure-rate/budget sensitivity 回答。

---

## 4. T3：KV、shadow 与 output correctness

### 4.1 KV 状态机

对每个 immutable content group，状态为：

```text
ABSENT -> RESERVED -> WRITING -> COMMITTED -> RECOVERY_PINNED
             |           |          |               |
             +-> REVOKED <-+---------+               v
                                           ADOPT_PREPARED -> ADOPTED
                                                  |
                                                  +-> RECOVERY_PINNED (abort)
```

- content identity：`(security_domain, compatibility_hash, prefix_hash, chunk_id, shard_id)`；
- copy lifecycle：`(source_execution, source_block_generation, destination_handle, destination_generation, action_id, creation_epoch)`；
- recovery lifecycle：`(recovery_execution, route_version, action_id, issuer_epoch)`。

`compatibility_hash` 与 `prefix_hash` 有不同责任。前者覆盖所有会改变 KV tensor ABI 或值的执行语义，包括精确 base/adapters revision、KV dtype/codec、position/RoPE/mask semantics、layout/sharding/page format 和 multimodal encoder revision；后者覆盖 canonical token/position/mask inputs 以及 request-specific multimodal/embedding digest。两者都匹配仍只证明 KV 可解释，不证明 sampler、grammar 或 tool parser 可以继续；后者由 Router 的 durable continuation-state 单独负责。

digest 仍不等于 recoverability。令一个 `N`-bit image/tool-result payload 只存在于 source worker，而 Router 只持有固定长度 digest 或指向 source-local/临时位置的 reference。source fail-stop 后，digest 不能作为 retrieval channel 重建任意 payload，reference 也不再解析；目标因而不能重建 canonical model input 或 silent replay。E1 必须让 actual payload（或完全等价、版本兼容的 embedding）以 content-addressed、授权且跨 source failure domain 的 object 存活。只有 hash 而没有可解析 object 时必须 retry。这个必要性由信息缺失直接推出，不需要性能实验。

### 存储提交引理（per-object two-phase put 与 group ID 不推出整组原子可见）

令一个最小恢复 group 需要两个 Store objects `K,V`。底层 `PutStart/PutEnd` 对每个 key 独立完成，batch API 也返回逐 key 结果。存在合法执行：`K` 已进入 `COMPLETE`，`V` 的 transfer 或 `PutEnd` 失败。即便两个 key 使用相同 group ID，group routing、共同 lease refresh 和协同 eviction 也只约束已经存在的成员，并没有产生一个“`K,V` 同时变成可读”的线性化点。任何仅凭 `K` 存在、枚举 component keys，或把 batch call 返回当成单一布尔值的恢复协议，都可能把 partial group 加入 `V`。

Aegis 对 Store 使用 marker-last 规则：先以不可变 key 完成全部 required component objects；验证逐 object checksum、rank-local generation、compatibility/content/lifecycle identity 后，最后用同一普通 object API 写一个小型 immutable **commit record**，其中列出精确 member keys/checksums/generations。恢复端只从 commit record 发现 group，而不从 raw component existence 推断；`AcquireGroupReadLease` 在一个 Master metadata transaction 中重新验证 record 与全部 named members并原子增加所有 read references，任一失败整体 rollback。lease 同时覆盖 record 与 members，直到 bounded load terminal release。record 不存在时 group 不可见；record 存在但任一 member 缺失、checksum 不符或 lease 失败时，整个 group 被排除并把连续 prefix 截断到前一组。因此 partial per-key completion 或 validation-to-pin eviction race 永远不能产生 recovery value。

**证明。** 写 record 的前置条件包含全部 required members 已完成且身份一致，所以 record 发布之前没有协议允许的 `ABSENT/WRITING -> COMMITTED` 转移。发布之后，恢复端唯一的发现入口先读取 record 给出的有限 member set；Master transaction 把最终全组验证与全组 read-reference 获取线性化为一个 transition，失败时不留下 partial lease，成功后又把 record/members 的 reclaim 排在 terminal release 之后。故任一允许的读取只会得到完整 group；底层出现 partial success、迟到 ACK 或 marker/member eviction 时至多损失可用性，不会错误恢复。这个结论是协议安全性质，不需要 workload 或性能实验；实现只需用 partial-write、marker-before-member 拒绝、missing-member、atomic group-lease-vs-evict 和 checksum fault tests 验证没有旁路。

`COMMITTED` 只把 group 加入 protection-visible set `V`。failure plan 必须再通过 two-phase group `FreezeAndPin` 把一个连续 prefix 加入当前 recovery-visible set `V_R`：各 rank 先 `PIN_PREPARED`，只有所有 required ranks 都成功时 coordinator 才发布 `PIN_COMMIT`，否则整体 abort并释放 prepared refs。只有 `RECOVERY_PINNED` group 可被 load/adopt。peer shadow 只有到 `ADOPTED` 后才进入普通 allocator/radix 与 Conductor GPU visibility。

### Peer-HBM 发布引理（本地 page-table 写入不等于请求可路由）

目标 rank 可以在不同时间完成本地 D2D 或 page-table binding，所以不能假设多块 GPU 上存在物理同时的“原子写”。Aegis 把原子性放在更窄、也可实现的准入点：shadow page 从 reserve 起已经移出 allocator free set，但在健康路径中没有 `req_to_token`/radix mapping；故障后每个 rank 在同一 recovery pin 下重新检查 `destination_handle/generation`，完成本地 mapping 或 D2D 后进入 `ADOPT_PREPARED`。此时 recovering request 仍在 quarantine queue，既不进入 scheduler ready/running set，也不发布普通 GPU placement。只有 coordinator 收到全部 required ranks 的 prepare ACK，才发布 versioned `ADOPT_COMMIT` 并返回唯一 route permit；Worker 收到 permit 后才把 request 加入 runnable queue，Router 也只接受匹配该 permit 的 continuation。任一 prepare 失败都删除已写的临时 mapping；D2D 版本释放新 primary pages但保留 recovery-pinned shadow，zero-copy 版本恢复 shadow ownership，二者都不产生 route permit。

**证明。** 在 `ADOPT_COMMIT` 之前，隔离前提使 request 不属于任何可调度/可路由集合，因此即使部分 rank 已有合法本地 mapping，也不存在一次 forward 能观察到 mixed-rank state。`ADOPT_COMMIT` 的前置条件逐 rank 检查相同 recovery execution/route、完整连续 group、仍匹配的 destination generation 与本地 terminal ACK；因此 commit 之后每个 required rank 都有同一 prefix boundary 的有效 mapping。route permit 是唯一把 request 加入 runnable set 的 capability，所以任意可执行 forward 要么发生在 commit 前而无法看到该 request，要么发生在 commit 后而看到全部 rank 的合法 mapping。partial prepare、stale generation 和迟到旧-route ACK 至多导致 rollback/重算，不会产生 partial routability。这个安全命题不需要 GPU 性能实验；实现只需以 generation-reuse、partial-rank prepare、rollback、permit-before-enqueue 和 failure-after-local-map 的确定性 fault tests检查 premise。

### 定理 2（不会恢复 partial、stale、已回收或 speculative KV）

假设唯一的 `COMMIT` transition 同时检查：

1. 全部 layout-required shards/stages 对同一逻辑 prefix、token range、source execution、action 和 creation epoch ACK；每个 shard 的 byte checksum、source generation 与 destination handle/generation 单独记录，不要求不同 rank 的实际 KV bytes/generation 相同；
2. 每个 source shard 的 versioned read pin 持续到 copy/checksum ACK，且其本地 execution/generation fence 有效；
3. checksum、security domain、完整 compatibility/prefix identity 和 failure-domain placement 有效；
4. group 紧接当前 continuous committed prefix，且不越过 source committed boundary。

再假设 `FreezeAndPin` 与 target 上的 commit/reclaim 使用同一线性化顺序，并同时检查：

5. 全部 required ranks 对当前 `(recovery_execution,route_version,issuer_epoch)` prepare pin，且唯一 coordinator group-commits共同连续 prefix；任一失败会整体 rollback；
6. pin 把引用从可过期 protection lease 转入 recovery lifecycle，直到 adoption/abort terminal ACK 才能释放；
7. 实际采用的 group end 不超过 Router write-ahead log 中 `last_durable_output_sequence` 给出的 `durable_continuation_token_end`；跨过该 boundary 的 chunk 整体退到前一个完整 group。
8. peer adoption 在每个 rank 重新验证 destination generation，本地 prepare 期间 request 不可调度；只有 all-rank `ADOPT_COMMIT` 产生 route permit，partial prepare 整体 rollback。

则 `V_R` 中任一 prefix 完整、版本一致、授权正确、无 hole、未被回收，且不含未持久化的 speculative suffix；恢复只读取 `V_R`，peer request 又只在 all-rank route permit 后可运行，所以不会使用 partial、stale、use-after-free、未接受或 partial-adopted KV。

**证明。** 对状态转移数归纳。初始 `V,V_R` 为空，性质平凡成立。reservation、write、partial ACK、revoke、late ACK 和 stale-generation event 都不改变 `V`。唯一增加 `V` 的 commit transition 显式检查 1–4；条件 1 比较共享 group identity，同时保留并验证 rank-local certificate，条件 2 防止 copy 读取被回收/复用的 source page。每个 rank 的 pin prepare 把本地并发 commit/reclaim 排在 freeze 前或后，但 prepared state 仍不进入 `V_R`；只有 coordinator 在全部 ranks 通过 5–7 后提交的共同连续 groups 才进入 `V_R`，任一失败只删除 prepared refs。进入后条件 6 阻止 expiry/reclaim；本地 `ADOPT_PREPARED` 仍不增加 runnable set，只有满足条件 8 的 commit 才转移 ownership 并发布 route permit，abort 只删除临时 mapping而保留 pin。故每次 transition 后结论保持。证毕。

### 必要反例：KV compatibility 与 continuation state 不能互相替代

第一，给两个请求相同 token IDs、model name 和 layout，但让一个启用 adapter `A`、另一个启用 adapter `B`，或让同一个 image-placeholder token 指向两个不同图像 embedding。若 identity 没有覆盖 adapter/side input，两者 key 相同而 KV 值不同；恢复后使用前者 KV 继续后者请求在语义上错误。第二，固定完全相同的合法 KV 与已接受 token prefix，让两个 continuation 分别处于不同 RNG counter 或 JSON-grammar FSM state；它们下一步允许的 token distribution 不同。故 KV 完整性无法推出请求 continuation 完整性。

这两个反例已经证明 identity 字段的必要分离，不需要模型吞吐或故障性能实验。实现责任只是验证版本化 identity、序列化/重建 continuation-state、unsupported state 的 fail-closed 分支，以及 speculative draft state 不进入 authority。

### 必要反例：single-sequence record 不能恢复 multi-hypothesis search

考虑两个 beam-search executions，它们持久化的当前 top hypothesis、KV 和 RNG record 完全相同，但第二个 live hypothesis 的 token、score 或 parent 在两者中不同；该第二 beam 在后续一步可能分别超过或不超过当前 top。只恢复 top sequence 无法区分这两个合法状态，最终选择可能错误。`best_of>1`、joint reranking 与其他有竞争 hypotheses 的算法有同样问题。

要透明恢复此类请求，必须原子持久化全部 live hypotheses 的 token/KV references、scores、finished flags、parent relation 和每条 continuation-state，并给它们共同的 durable boundary。Aegis E1 首版不承担这个复杂度，只证明单一 target-accepted authoritative sequence；multi-hypothesis 请求没有 backend 提供的完整 bundle 时 fail closed/request retry。该 scope 由状态不可区分反例决定，不需要性能实验。

### 推论 1（Controller failover 不应作废 immutable commit）

creation epoch 证明哪个 leader 创建了 commit certificate；issuer epoch 决定谁有权发出当前 mutation/recovery command。若把二者混成“必须等于 current epoch”，leader change 会让所有仍完整的 peer copies无故失效；若完全不检查 epoch，旧 leader又能继续续 lease或 revoke。正确规则是：新 leader重建并验证旧 creation-epoch certificate 后，可以用更高 issuer epoch pin/adopt它；所有旧 issuer-epoch mutation 被拒绝。该规则同时给出 failover 可用性与 single-writer fencing。

若 leader change 发生在 `RECOVERY_PINNED` 之后，pin 不能因 control failover 自动释放，也不能被新 leader仅凭单个 rank report 接管。所有 required ranks 必须对同一旧 recovery execution、route、issuer 和 group set 完成一次 all-rank `RebindPinSet`，才把 mutation authority 转到新 issuer；partial rebind rollback，旧 capability 继续只允许原计划走向 terminal state。若在有界 watchdog 内无法重建，则各 rank 协调 abort 并退化到 Store/recompute。由此既不会让旧 leader释放新 leader 正在读取的 pages，也不会形成永久 pin leak。

### 推论 2（未 adoption shadow 不可路由）

`COMMITTED` 只证明 private shadow bytes 完整，不证明目标 request page table 已绑定这些 pages。若 Conductor 在 adoption 前发布普通 GPU hit，Router 可能把 cache reuse 请求送到一个无法寻址的对象。单个 rank 的 `ADOPT_PREPARED` 也不够，因为其他 rank 可能还没有 mapping。故 GPU placement event 与 scheduler enqueue 必须发生在 all-rank `ADOPT_COMMIT` 的 route permit 之后；不支持直接接管时，D2D copy、双份 occupancy 和 barrier 必须计入 `p_j`。

### 推论 3（只有 sealed immutable chunk 可异步保护）

若 source page 在 copy/checksum 期间仍可被 decode kernel 修改，则不同 shard 或不同传输片段可能来自不同逻辑时刻，checksum 成功也不能证明 snapshot 一致。故首版只复制 sealed chunk；partial tail 没有 COW/snapshot barrier 时必须重算。

### 定理 3（条件化的逻辑 output exactly-once）

在 E1 中 Router 不失败。再假设：

1. Router 只接受匹配 current execution/route fence 的 worker output；
2. token 在首次外发前，以一个原子 Router WAL transition 写入 `(sequence, token_id, next_sampling_position, next_state_digest)`；记录的 hash chain 从有序 canonical input-object records、canonical request 与 initial state 开始，并覆盖 previous-record digest，最后一个 committed record 与 versioned continuation-state 的 log-tail/state digest 必须相等。若 state 可从 canonical request/token history 重建，digest 绑定该重建规则；否则 immutable state blob 必须先在 source failure domain 外 durable，record 再引用其 digest。torn tail 整体丢弃，不能出现“token durable 但 next state 不 durable”或相反；该 state 包含或可重建 sampling config、logit processors、remaining stop conditions、grammar/tool-parser state、active model/adapters 与可解析的 failure-surviving canonical input object references；
3. Router 的 durable ordered outbox 持久化 `last_externally_published_sequence`；outbox send、pointer advance 与 worker-failure handler 在同一个存活 Router event loop 中串行，`SENDING/PUBLISHED` 不会因 worker recovery 退回 `PENDING`，所以在 E1 的存活 stream enqueue boundary 上每个 sequence 最多提交一次；
4. token position 采用半开区间：canonical input 为 `[0,P)`、durable output 数为 `n` 时 continuation boundary 为 `B=P+n`；recovery 只使用 end 为 `K<=B` 的完整连续 KV group，跨过 `B` 的 speculative group 被截断；
5. 新 worker 在 `K<P` 时先 teacher-force canonical input `[K,P)`，再 teacher-force output sequences `max(1,K-P+1)..n`；两段恰好一次覆盖 `[K,B)`，同时重建 suffix KV且不产生网络输出；
6. silent replay 后完整 continuation-state 对齐 `last_durable_output_sequence+1`，new worker 只能从该 sequence 生成；speculative draft state 被丢弃，Router 先从 outbox 发送尚未 externally published 的 durable entries，再按序接受新输出；
7. tool sink 以 `(workflow,agent,turn,tool_call_id)` 幂等提交。

并且请求只有一条 authoritative target-accepted hypothesis；则 worker fail-stop recovery 不会在逻辑 token stream 中产生 duplicate/gap，不会重复提交已经记录的 tool call，也不会因丢失 sampler/grammar/adapter/side-input state 从错误的 request semantics 继续。未被 Router durable 接受的随机未来可以不同，本定理不声称 bitwise 重现该 counterfactual future，也不覆盖未持久化完整 hypothesis bundle 的 beam/best-of search。

**证明。** fence 排除旧 worker 的后续输出；条件 2 的原子 record 给每个 Router 接受的 token 唯一 durable sequence 与后继 continuation-state，hash chain 又把 record 的顺序、token、sampling position、state 和 canonical input 绑定。任一 torn/reordered/substituted tail 要么没有 commit，要么验证失败，不能成为 recovery boundary。条件 3 让“durable 但尚未发送”的窗口由串行 outbox state 唯一决定并排除 worker recovery 触发重发。条件 4 排除无法由 log 证明的 speculative state。对任意 `K<=B`：若 `K<P`，条件 5 的两段长度之和为 `(P-K)+n=B-K`；若 `K>=P`，其长度为 `n-(K-P)=B-K`。两段均按 canonical 顺序开始于 `K`、结束于 `B`，故恰好重建 KV gap 一次且不产生网络输出。条件 6 使完整 request state 严格接在 durable boundary 后，同时 outbox 先补齐尚未 externally published 的 durable entries。因此既不会跳过一个 Router 已接受 token，也不会在同一存活 stream 上重复提交一个已外发 token；兼容 identity 与 continuation-state 的共同检查排除上述两个反例。条件 7 把重复 tool attempt 映射到同一个 sink operation。证毕。

若 token 先外发、next state 后写，构造 worker 在两者之间 fail-stop：Router 已不能撤回 token，target 又没有该 token 后的 grammar/penalty/sampling authority，故透明 continuation 不成立。反过来只写 state 而没有 token，target 也无法知道要 silent replay 哪个 accepted token。这个二状态反例已经证明原子 WAL transition 的必要性；不需要做性能实验。hash chain 不是为了 Byzantine consensus，而是让 crash/torn-write/reorder 在恢复时 fail closed，而不是让一个脱离实际 token log 的 state digest 字符串冒充 authority。

对当前 pinned SGLang 的首版 mapping 也不需要抽象的“任意 RNG snapshot”。普通 repetition/frequency/presence penalty 由 accepted output IDs 累积，grammar 对 accepted token 调用 `accept_token`，stop/parser 状态由 canonical request 与 token/text prefix 重放。若启用 deterministic inference，采样使用 request `sampling_seed` 与绝对 sequence position；故 V1 continuation record 保存 sampling config/seed 与下一个 position，并要求每个单-token WAL record 的 position 连续递增。未请求 deterministic future 的普通随机采样可在已接受 prefix 后使用新随机未来，因为定理本来就不声称复现未被 Router 接受的 counterfactual token。任意 custom logit processor、非标准 position 规则或无法由 token history 重建的 sampler 必须提供版本化 serializer/replayer，否则 fail closed/request retry。

这不是任意网络下的 transport exactly-once，也不证明客户端消费 side 的 exactly-once：若 stream 断开、Router 也失败或 client reconnect，必须使用 client ACK/sequence dedup；Router failure 还需要复制 WAL，否则降级为 request retry。

### T3 的证据责任

状态机证明给出 safety；property/fault-injection tests 检查实现没有旁路 transition；actual-byte checksum 与 process hard kill 证明真实 data path 符合状态机。吞吐图不能替代三者中的任何一个。

---

## 5. T4：EDF、release/deadline 分离与 chunk bound

### 定理 4（固定版本区间上的理想 preemptive EDF）

在固定 target/route/due-date/service-envelope version、一个串行 machine、独立 jobs、已知 release/deadline、无代价任意抢占下，EDF 最小化 maximum lateness，即 `1|r_j,pmtn|L_max`。这是经典结果，Aegis 只引用 [Horn 1974](https://doi.org/10.1002/nav.3800210113)，不把它作为新算法。

当所有 release 相同且 jobs 完全非抢占时，按 due date 的 EDD 是 Jackson 特例。

### 必要反例：动态收紧 deadline 不保留 whole-history optimality

jobs A、B 都在时刻 0 释放，processing time 都为 1。初始 due dates 为 `D_A=100,D_B=2`，所以 online EDF 在 `[0,1]` 执行 B。时刻 1，Algorithm 1 的新版本把 live job A 的 due date 收紧为 1；online schedule 随后完成 A，得到 `C_A=2`、`L_max=1`。但一个预知最终 due dates 的 clairvoyant schedule 会按 `A,B` 执行，得到 `C_A=1,C_B=2`、`L_max=0`。即使 deadline 只会单调收紧，过去的顺序也无法撤销，因此 Aegis 不能把 Horn 的 fixed-instance 结果扩张成相对于 future-aware full-history optimum 的保证。

正确 comparator 是条件后缀：route、live due date 或 service-envelope version 改变时，在更新线性化点固定已经执行的 prefix、当前 remaining work 与新版本输入；EDF 只与共享该 executed prefix 的合法 suffix schedules 比较。若更新发生时已有一个不可抢占 quantum，`ISOLATED_EDF` 可把它的剩余 terminal time 作为至多一个小于 `B_g` 的初始 blocking；若旧 service contract 不再保证它完成，或新 envelope 不能界定其 residual time，则新 interval 不能进入 theorem mode。普通新 job release 不需要重建历史，因为 release-aware EDF 本来就允许动态 arrival；改变 live job 的 deadline/service contract 才改变已经发布的调度 instance。

这个反例与 comparator 修正是纯理论结论。实现责任只是：due/service version 原子更新 ready queue 和 remaining-work bound；弱化旧 service contract 前保住已发 quantum 到 terminal，或 fail closed；telemetry 报告 version changes。无需用 workload 性能实验重新证明不可能性。

### 反例（EDF 不优化 lexicographic risk 的第二项）

`T_max=max(0,L_max)` 是 `L_max` 的单调函数，所以 EDF 也最小化 maximum tardiness；但这个定理到第一项为止。考虑三个在时刻 0 释放、权重相同的 jobs：

| job | processing | due date |
|---|---:|---:|
| A | 1 | 2 |
| B | 2 | 3 |
| C | 3 | 1 |

EDF/EDD 顺序 `C,A,B` 的 tardiness 是 `(2,2,3)`，故 `(T_max,sum T)=(3,7)`；合法顺序 `A,C,B` 的 tardiness 是 `(0,3,3)`，故 `(3,6)`。两者第一项相同，后者第二项更小。因此 EDF 不是 `(maximum tardiness, total tardiness)` 的 lexicographic optimizer，`shortest remaining service` tie-break 也没有这样的全局保证。

Aegis 只引用 EDF 对第一项的 optimality。第二项是在固定 EDF dispatch policy 下评估 protection actions 的系统指标；其最优性或 regret 交给离线 oracle，而不是扩张 EDF theorem。

### 必要反例：target availability 不能成为 deadline

failure cutoff `F=0`，UNKNOWN active job 的 target 在 `R=3` 才可用，service 为 1。正确 deadline 是 `D=0`，所以 completion `C=4`、lateness 为 4。若错误设成 `D=R=3`，lateness 变成 1，凭空隐藏 3 秒 target waiting。

### 必要反例：idle-time `DEFERRED` work 不是零干扰

时刻 0 没有 foreground job，executor 启动一个不可抢占、长度 `B_g=5` 的 deferred/background quantum。时刻 `epsilon` 到达一个几乎零 service、deadline 也为 `epsilon` 的 urgent job。即使规则在每个 dispatch boundary 都让 foreground 优先，urgent job 仍只能在时刻 5 后完成，因而被 background 阻塞接近 `B_g`。所以“只用 otherwise-idle capacity”不能推出零 foreground delay；它只说明 background 不会在**已有 ready job**前被新 dispatch。

若 `ISOLATED_EDF` 始终只有一个 outstanding quantum、其 terminal time 不超过 `B_g`，且 foreground ready set 非空后不再 dispatch 新 background quantum，则每个 foreground busy period 最多继承一个这种 background interference；它就是定理 5 critical-block 中的较晚-deadline quantum，因此仍由 `<B_g` additive term 覆盖。多个 outstanding operations、无 terminal bound 或 pipelined shared resources 都关闭这个推论。反例和推论不需要性能实验；实现只验证 dispatch priority、one-outstanding 与 terminal bound。

### 定理 5（严格 chunk executor 的 additive bound）

固定 target `g`。假设：

1. 只有一个 work-conserving logical recovery lane；
2. 每次只 dispatch 一个 job 的一个 quantum，不做 cross-job Store/NIC/GPU overlap；quantum 内部可以用多个资源并行，下面的 `p_k` 取整个内部 operation DAG 的 terminal makespan；
3. quantum 从 dispatch 到 terminal ACK 的总时间 `p_k<=B_g`；
4. quantum 完成前不抢占，但边界处选择 ready job 中最早 deadline；
5. 所有 dispatch/setup/barrier cost 已计入 `p_k`，没有取消后丢弃并重做的 work；
6. 一个 job 的所有剩余 work 都在原始 `R_j` 时释放；quanta 只定义不可抢占粒度，不生成新的 release time。每个 quantum 继承原 job 的 release/deadline，只要 job 未完成，边界处就总有它的 next quantum eligible。

令 `L_{\max}^{\mathrm{chunk}}` 为该 chunked EDF schedule 的 maximum lateness，`L_{\max}^{\mathrm{pmtn}}` 为同一 jobs、release、deadline 与总 processing work 的理想 preemptive optimum，则

\[
L_{\max}^{\mathrm{chunk}}<L_{\max}^{\mathrm{pmtn}}+B_g.
\]

**证明。** 把每个 recovery job 看成若干 processing quanta；同一 job 的全部 quanta 在 `R_j` 后都代表已经释放的 remaining work，且带相同 deadline。它们的内部编号不影响 job completion。该 schedule 是 ready quanta 上的 extended Jackson/Schrage rule。这里没有把第二个 quantum 的 release 错设成第一个 quantum 的 completion，因此理想 comparator 没有被人为加上 precedence。

取 job-maximum lateness 的 final critical quantum `c`。若其所在 critical block 中没有 deadline 晚于 `D_c` 的 interference quantum，则该 block 达到 release/work demand lower bound，schedule 对该 block 已最优。

否则令 `b` 为 `c` 前最后一个满足 `D_b>D_c` 的 interference quantum，并令 `K` 为 `b` 后直到 `c` 的连续 quanta。EDF 在 `b` 开始时选择了较晚 deadline 的 `b`，所以 `K` 中任何未完成 job 当时都不可能 ready：若它已经在原始 `R_j` 释放且尚未完成，边界处必有它的 next quantum 可执行，并应优先于 `b`。因此 `\min_{k\in K}R_k>s_b`。同时 `D_k<=D_c`。若某 schedule 的 maximum lateness 为 `L`，则每个 `k\in K` 都必须在 `D_k+L<=D_c+L` 前完成。又因为这些 work 都不能早于 `\min_{k\in K}R_k` 执行，故任意 schedule，包括理想 preemptive schedule，都满足

\[
L_{\max}^{\mathrm{pmtn}}
>s_b+\sum_{k\in K}p_k-D_c.
\]

chunked schedule 中

\[
L_{\max}^{\mathrm{chunk}}
=s_b+p_b+\sum_{k\in K}p_k-D_c.
\]

两式相减得

\[
L_{\max}^{\mathrm{chunk}}-L_{\max}^{\mathrm{pmtn}}<p_b\le B_g.
\]

证毕。这个 critical-block 论证与 extended Jackson/Schrage 的经典 absolute-error bound 相同；可参考对该 bound 的整理与证明：[Kacem and Kellerer, Algorithmica 2018](https://doi.org/10.1007/s00453-018-0417-6)。

### 定理 5 何时立即失效

以下任一项出现，就只能声明 heuristic EDF 并报告 oracle regret：

- 多个 outstanding transfers；
- cross-job pipeline overlap；
- 一个 batch 混合多个 deadline；
- staging path 无有限 terminal ACK/cancel bound；
- cancel 丢弃 work 后重新传输；
- setup/queue switching 没有计入 `B_g`；
- shared resource 没有 isolation，导致一个 lane 的 quantum duration 依赖其他 lanes。

因此定理减少的是“需要跑多少 EDF 算法实验”，不是免除 executor premise validation。microbenchmark/telemetry 仍要确认部署是否有资格标记 `ISOLATED_EDF`。

---

## 6. T5：多资源量纲与容量安全

### 引理 6（normalized demand 必须同量纲）

对资源 `r` 定义 `u_r(a)=Demand_r(a)/Budget_r`。只有 numerator/denominator 同单位时 `u_r` 才是无量纲 utilization：

- HBM/Store：`bytes / bytes`；
- window traffic：`bytes / (bytes_per_second * window_seconds)`；
- source pin：`byte_seconds / byte_seconds`；
- reservation：`count / count`。

把 one-shot bytes 直接除以 rate 得到 seconds，再与 byte utilization 取 `max` 没有物理意义。lead time/deadline 是单独 feasibility gate。

### 定理 6（peer HBM 不侵占声明的 primary reserve）

设 primary growth-rate、peer reclaim/cancel terminal latency、单次 allocator
burst 和 fragmentation 的 hard upper bounds 分别为
`dot U_primary^+`、`T_terminal^+`、`Burst^+`、`Frag^+`。定义

\[
Reserve^*=\left\lceil
\dot U_{primary}^{+}T_{terminal}^{+}+Burst^{+}+Frag^{+}
\right\rceil.
\]

任一有限 hard bound 缺失时，theorem-mode peer budget 为 0。对每个 required
rank，令 `U_peer` 包含 committed、reserved/in-flight、recovery-pinned 的 source
shadow occupancy；D2D 新分配的 quarantined destination page 进入 `U_primary`。
`d2d_pending` 只标记同时被锁住的 source/destination 子集，不作为第三份 bytes。
维护：

\[
U_{primary}+U_{peer}+Reserve_{remaining}\le C_{KV}.
\]

假设：

1. 所有 primary allocation、peer commit、adoption、release 和 capacity change 都经过同一原子 ledger；
2. NORMAL 状态有 `Reserve_remaining=Reserve^*` 且没有 pending D2D，peer admission 只在 transition 后仍满足不等式时成功；
3. primary growth 可等量增加 `U_primary`、减少 `Reserve_remaining` 并进入 DRAINING；DRAINING 禁止新 peer admission，超过 remaining reserve 的 allocation 在 commit 前阻塞/回收或交给 baseline OOM policy；
4. generation-fenced terminal reclaim 减少 `U_peer` 并补回 `Reserve_remaining`，达到 `Reserve^*` 才回到 NORMAL；pin/in-flight bytes 始终在 `U_peer` 中计费，不在 Reserve 中重复计费；
5. capacity shrink 在 peer reclaim 后仍满足新容量不等式时才 commit；若 `U_primary>C_KV^{new}`，属于 baseline capacity failure，不在本定理内；
6. D2D adoption 在同时占有 source shadow 和 destination primary page 时，前者留在 `U_peer`、后者进入 `U_primary`；begin 只允许从 NORMAL 发起，并要求临时 destination `q` 满足 `U_primary+U_peer+Reserve^*+q<=C_KV`，不减少 `Reserve_remaining`；pending 状态禁止新 peer admission，且 `d2d_pending<=min(U_peer,U_primary)`；
7. emergency transition 只释放 peer state，不删除 primary state。

则在声明的 primary-growth/reclaim envelope 内，Aegis 的 committed transitions 不会超过物理 KV capacity，也不会把 NORMAL peer admission 放进声明的 primary headroom。

**证明。** 对 ledger transition 归纳。初态满足不等式。NORMAL peer admission 由条件 2 保持不变量；release 只减小左侧。primary growth 的 `+Delta U_primary` 与 `-Delta Reserve_remaining` 抵消，且 remaining reserve 不足时不允许 commit；terminal reclaim 的 `-Delta U_peer` 与 reserve replenishment 同步，补回不超过 `Reserve^*`。capacity shrink 由条件 5 在新右端上重检。zero-copy adoption 把同一 bytes 从 `U_peer` 移到 `U_primary`，总和不变；D2D begin 保留 source 的 `U_peer` charge，并由条件 6 只使用完整 reserve 之外的 headroom，故增加 destination 后不变量仍成立且 safety reserve 未缩短；finish 释放 source，abort 释放 destination，若期间普通 primary growth 已合法消费 reserve，释放出的物理空间还可把它补回但绝不超过 `Reserve^*`。`d2d_pending` 不作为第三份 occupancy，只阻止任一端被提前回收并关闭新 peer admission。emergency 只减小 `U_peer`。所以每次 committed transition 后不变量成立。证毕。

该定理不声称 baseline primary 自身永不 OOM，也不保证零 allocation stall，更不证明 feedback latency、TTFT 或 goodput。特别地，若一个不可再分的 D2D quantum 放不进完整 `Reserve^*` 之外的 headroom，P0 D2D edge 在该 snapshot 下就是不可行，而不是允许借用 safety reserve；Planner 必须改用 Store/recompute/zero-copy，或仅在每个 chunk 都可独立完成 source release、所有 chunks 在最终 route permit 前保持 quarantine 时使用 bounded-chunk D2D。这个结论属于安全与可行性分析；后者的性能幅度仍需少量 step response 和 end-to-end measurement。

`alpha` 的新 cap 也不是该定理的瞬时 occupancy 不变量。若 cap 降到当前 `U_peer` 以下，正确 transition 是进入 DRAINING、禁止新 admission，并只在 generation fence 与 terminal ACK 后减少 `U_peer`；recovery pin 可以暂时阻止到达新 cap。由归纳仍可证明物理 capacity invariant，因为 pinned/in-flight bytes 没有消失且 Reserve 继续计费；但“多久回到新 cap”不能由一次 actuator assignment 推出，必须报告 cap overdraft 与 step response。

### 推论 3（三层 service contract）

对每个实际 resource `r`，若

\[
F_{\mathrm{serve},r}+F_{\mathrm{recover},r}
\le C_{\mathrm{effective},r},
\]

background protection 只使用 residual token bucket，且 failure 时可降到 0，则 Aegis 不会通过 background copy 借用声明的 normal-serving floor。这个结论依赖 admission/shaping，而不是 AIMD 最终收敛；AIMD 只决定 residual capacity 利用速度。

### 引理 7（单 actuator 的 persistent-signal response bound）

令 `x_0` 位于 `[x_min,x_max]`，overload update 为
`x_{n+1}=max(x_min,beta*x_n)`，其中 `0<beta<1`。对任意
`x_min<=theta<x_0`，若 overload signal 持续且无反馈延迟，则在至多

\[
N_{down}=\left\lceil\frac{\log(\theta/x_0)}{\log\beta}\right\rceil
\]

次 update 后有 `x<=theta`。若 `x_min>theta`，普通 loop 不可能达到该阈值，必须由 DRAINING/emergency override。反之，若每累计 `K` 个稳定窗口执行一次 `x<-min(x_max,x+Delta)`，则对 `x_0<theta<=x_max`，达到 `theta` 至多需要

\[
N_{up}=K\left\lceil\frac{\theta-x_0}{\Delta}\right\rceil
\]

个稳定窗口。

**证明。** 在 clamp 生效前，第 `n` 次 decrease 后 `x_n=beta^n x_0`；解 `beta^n x_0<=theta` 即得第一式。increase 每次至少增加 `Delta` 直到 cap，所需 update 数为第二式括号内项，每个 update 按定义需要 `K` 个稳定窗口。证毕。

该引理不证明 delayed/noisy feedback 下的 closed-loop stability，也不覆盖 memory/rate 两个 actuator 的 cross-coupling。那些真实系统效应才需要少量 step-response measurement；clamp 与 persistent-signal 公式无需用实验重复证明。

---

## 7. 论文中应如何使用这些结论

| 结论 | 主文写法 | 需要的非理论证据 |
|---|---|---|
| T1 不可辨识性 | 一个反例 + fail-closed 设计 | 无；G1 只测 opportunity |
| hard Slack/expiry | 引理 1–2；bounds 明确量化 failure-conditioned fastest-sibling / slowest-target continuation，且不得循环借用 sibling recovery wait | witness horizon + invalidate-before-effect + service-envelope enforcement tests |
| cutoff-normalized due date | `D_i^F=max(F,deadline_at)` 只冻结正 remaining Slack；零/过期 ticket 使用 `D_i^F=F`，引理 2a 给出 path-shift chain | terminal ACK 覆盖 `F` 后全部 failure-only catch-up；无需性能实验 |
| exact action 三段式与反例 | 一张三段函数图或小表 | G0 测真实 disagreement prevalence |
| post-commit action value 不可由 current snapshot + copy bound 精确识别 | 反例 3；无 job-liveness 与 prefix/service horizon contract 时明确标 myopic snapshot score | 完整 contract 对允许状态取最小 risk reduction；否则只在 commit 时重算 |
| soft Store priority 不能证明未来最低驻留层 | 反例 3a；deterministic state 只接受可拒绝、quota-backed、覆盖 horizon 的 tier lease | Master quota/placement/expiry/failover conformance；无需性能图重证不可辨识性 |
| Recovery Planner composition | 引理 5a 的 plan-set/null-option 单调性；串行 grammar 才使用 token-boundary shortest-path 命题；反例 4 排除一般并行等价性，反例 4b 排除拼接独立 per-job makespan | selected-set/joint-schedule conformance、operation eligibility 与 terminal-bound tests；measurement 只报 bound tightness/fallback 和 heuristic gap，不重证集合关系 |
| bounded score 是 deadline-risk upper-envelope reduction、不是 realized-gain lower bound | T2 单调性与数值反例 | G4 paired faults 测真实 WCT gain |
| absolute `C-D` 与 realized WCT 的等号范围 | T2 common-origin identity 与两-branch stale-origin counterexample | measurement 只报 risk tightness/realized conversion，不重证反例 |
| frontier `max` | 引理 4 + antichain 条件 | DAG frontier validator |
| exact protection selection 的 NP-hardness | T2 从 0--1 knapsack 的归约 | 无；G6 oracle replay 只测当前 greedy 的实际 gap |
| weight-free minimax scope | 两域 hazard 反例；不是 expected-loss 或 fairness optimum | canonical domain IDs 与 quota-pool cap conformance；具体 economics 只做 sensitivity |
| commit/pin/output safety | 定理 2–3：freeze-and-pin、creation/issuer epoch、durable continuation boundary | actual-byte race injection、speculative-output 与 hard-kill tests |
| KV identity 与 continuation-state 必须分离 | adapter/multimodal 与 RNG/grammar 两个反例 | identity/serialization/fail-closed conformance；无需性能实验 |
| digest/reference 不等于 side-input recoverability | source-local payload 丢失反例；E1 要求 failure-surviving canonical input object | object resolution/auth/version 与 hard-kill conformance；无需性能实验 |
| single-sequence record 不覆盖 beam/best-of | 两个相同 top record、不同第二 hypothesis 的不可区分反例 | 首版 multi-hypothesis fail-closed；若未来支持则验证完整 bundle 原子性，无需性能实验 |
| ideal/chunked EDF | 定理 4–5；只覆盖 maximum lateness/tardiness 第一项 | executor premise validation；secondary regret 用 oracle，不是重复验证定理 |
| dynamic deadline/service update | 两 job tightening 反例；每个固定版本只给相同 executed prefix 下的 suffix guarantee | versioned queue-update、remaining-work rebind 与 in-flight contract fence；无需性能实验 |
| deferred idle stealing 不是零干扰 | arrival counterexample；isolated mode 至多一个 `<B_g` background block | dispatch/one-outstanding/terminal-bound conformance；无需单独性能矩阵 |
| capacity/headroom safety | `Reserve^*` 解析式 + 定理 6 的 NORMAL/DRAINING 归纳 | hard-envelope enforcement/allocator transition property tests；step runs 只测 stall/响应速度 |
| desired peer cap 与 actual occupancy | cap decrease 触发 DRAINING；pin/in-flight 到 terminal 前不减账 | state-transition conformance；overdraft duration 才需 step measurement |
| ideal AIMD response | 引理 7 的 decrease/increase window bound | delayed/noisy/cross-coupled step response；不重测代数式 |

这些 proof obligations 全部通过，也不能替代 G0/G1 的现实机会、G4 的 end-to-end gain 或 G5 的 no-fault cost。反过来，性能结果再好也不能修补 T3 的 partial-state/output correctness 漏洞。
