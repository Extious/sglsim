# 08. 图表与预期结果

本节将论文主线映射到图表。结构参考 `One Pool - Two Caches`：先证明双重压力和动态最优点，再展示系统设计，最后展示整体结果、消融、敏感性和 scale。

已生成资产：

- 图表预览：[`../../figures/gallery.md`](../../figures/gallery.md)
- 图表 manifest：[`../../figures/manifest.json`](../../figures/manifest.json)
- 可编辑机制图：[`../../figures/src/`](../../figures/src/)
- Markdown 预览图：[`../../figures/png/`](../../figures/png/)

当前证据状态：

- Fig. 4, Fig. 5, Fig. 5b, Fig. 6, Fig. 7, and Fig. 7b are design-ready mechanism diagrams.
- Quantitative result plots now use preliminary SGLSim CSVs from the 5-seed
  hard-failover replay matrix, direct replay-derived counterfactuals, a
  DP=2/4/8 scale sweep, and a matched DP=4 peer-route replay pair. They are
  useful for paper layout, story validation, and early internal review, but
  remain below final evidence.
- Small-scale SGLang smoke runs on `hkbugpumanode` now validate the real
  serving path and matched Fig. 11 overhead datapoints. Jobs `32861`, `32863`,
  `32869`, `32870`, `32871`, `32874`, and `32899` form the current matched
  DP=2 Qwen3-8B no-failure overhead aggregate: HiCache write-through mean
  overhead averages 0.22% with range -2.23% to 3.05%, p95 overhead averages
  -0.19% with range -4.09% to 2.02%, and p99 overhead averages 0.31% with
  range -3.93% to 3.74%.
- SGLang failure-control-path smoke jobs `32977` and `32978` now validate the
  basic real SGLang DP=2 failover path: 4/4 routed in-flight requests survived
  `/simulate_gpu_failure` in both runs, 4 requests were captured and
  re-dispatched, all 4 completed responses carried `sglext.failover`, and
  `/simulate_gpu_recovery` emitted `gpu_recovered`. Job `32978` captured 760
  pre-failover output tokens, but both smokes still had 0 pre-failover
  backed-up tokens. This is retry-continuity evidence, not final Host-visible
  or peer-copy evidence.
- SGLang storage-backed failure smoke job `32980` enables
  `hicache_storage_backend=remote_backup` with a centralized remote backup
  server. It again completes 4/4 failover-retried requests and reports 5,132
  pre-failover backed-up tokens, making it the first real SGLang Host-visible
  frontier smoke. It remains below final evidence because it is not a paired
  Host-visible/Host-invisible policy comparison.
- The posthoc paired artifact `failure_visibility_pair_32978_32980` now ties
  the no-storage job `32978` to the remote-backup job `32980`: frontier tokens
  increase from 0 to 5,132 backed-up tokens, or 1,283 backed-up tokens/request,
  while p99 latency is 1.108x the no-storage control path. This is supporting
  evidence for Fig. 11b/Fig. 15, not a final policy-latency point.
- Follow-up SGLang probes narrow the restore gap. Job `32981` disables
  request-aware remote-backup retention and still observes `data_pages=0` on
  the retry path, ruling out lease GC as the primary miss cause. Job `32982`
  uses a single request and validates a real storage match/fetch:
  `data_pages=1297`, `Prefetch ACCEPTED`, and `prefetch_get` hits. Job `32983`
  then runs the same page-0 single-request shape with
  `hicache_storage_prefetch_policy=wait_complete`, completing prefetch IO with
  `completed=1297 / expected=1297` and admitting retry prefill with
  `#cached-token: 1297`. Job `32984` adds the matching no-storage
  single-request control; `failure_visibility_pair_32984_32983` records a
  1,297-token Host frontier delta and 1.124x visible/no-storage p99 latency.
  Job `32985` shows that the four-request shape still has `data_pages=0` even
  with `wait_complete`, so shared-prefix split restore remains open. Job
  `32987` then enables `UNIQUE_REQUEST_PREFIX=1` and proves that four
  concurrent retries can materialize remote-backup pages when each request has
  page-0 backed data: all four requests reach `data_pages=1301`,
  `completed=1301 / expected=1301`, and retry `#cached-token: 1301`. Job
  `32989` supplies the no-storage unique-prefix control; the paired artifact
  `failure_visibility_pair_32989_32987` records a 5,204-token Host frontier
  delta and 1.308x visible/no-storage p99 latency. Job `32991` applies the
  split-prefix storage fix and reruns the shared-prefix four-request shape:
  all four retry queries reach `data_pages=1297`, all four prefetch IOs
  complete `1297 / 1297`, and each retry prefill admits with
  `#cached-token: 1297`. The paired artifact `failure_visibility_pair_32992_32991`
  records a 5,188-token Host frontier delta and 1.210x visible/no-storage p99
  latency. Jobs `32993`/`32994` repeat the visibility pair with a longer shared
  prefix, extending the backed frontier to 20,548 tokens and materializing
  `#cached-token: 5137` on all four retries, but increasing p99 to 1.630x the
  no-storage control. Jobs `32998`/`32999` isolate the same long-prefix
  mechanism under matched normal write cost: both runs write 20,548 tokens,
  while the lookup-disabled run hides the Host frontier from retry and the
  lookup-enabled run materializes `#cached-token: 5137`; the visible path is
  still 1.487x slower. Jobs `33002`/`33003` rerun the same matched-write-cost
  shape with `best_effort`: all retry queries find `data_pages=5137`, one
  prefetch partially completes `2816 / 5137`, retry prefill observes
  `#cached-token: 2830`, and visible p99 is 0.966x the lookup-disabled control.
  These remain below final evidence because they are
  diagnostic smokes, not repeated paired AegisKV policy comparisons at larger
  scale.
- Large-scale SGLSim online candidate run `20260621_032719` completed 96/96
  request turns for 32 shared-prefix agents. It now emits Aegis frontier and
  candidate failover columns in `request_details.csv`: 24 critical-agent retry
  candidates, 100% Host-backed frontier at the candidate failure point, and
  13.45 ms mean candidate failover overhead. This is stronger than static
  templates, but still below final failure-injected recovery evidence.
- SGLSim failover replay smoke `20260621_042727` additionally sent 24 real HTTP
  retry replay requests after protected critical turns. It completed 120/120
  request rows with 24 rows marked `aegis_failover_replay=true` and 24/24
  replay rows dispatched cross-rank (`failed_source_dp_rank != assigned_dp_rank`).
  All 24 replay rows were bracketed by SGLSim hard failure control-plane calls:
  the source scheduler subprocess was terminated before replay and restarted
  after replay, and the rows are marked `aegis_hard_failure_injected=true`.
  All 24 replay rows also hit the runtime HostBackup directory
  (`aegis_host_backup_lookup_hit=true`), restoring 39,676 tokens from the
  directory lookup and recomputing 0 tokens. Of those protected tokens, 38,989
  prompt-prefix tokens were materialized as target-rank device-cache metadata
  before replay admission and appear as replay-row `cache_hit_tokens`.
  All 24 replay rows also found peer GPUReplica directory entries
  (`aegis_peer_gpu_lookup_hit=true`, 5,036 peer tokens); for this long-prefix
  workload the runtime lower-envelope still chose Host restore for all replay
  rows because peer GPUReplica would require recomputing most of the prefix.
  The scheduler slept for the replay restore plan with 13.51 ms mean observed
  restore sleep, so restore cost is now visible in online wall-clock latency.
  This still does not execute real CUDA tensor KV copy; it is a metadata page
  movement inside SGLSim.
- SGLSim failover replay matrix `sglsim-failover-replay-matrix-r5-20260621`
  ran 5 seeds and completed 600/600 rows with 120 hard-failure cross-rank
  replay rows. All 120 replay rows hit HostBackup and peer GPUReplica
  directories, restored 201,787 HostBackup tokens, materialized 198,322
  prompt-prefix tokens as target-rank cache metadata, recomputed 0 tokens, and
  observed 13.62 ms mean replay restore sleep. The current SGLSim measured CSV
  rows are derived from this r5 matrix as 5-run means with run-level std/95% CI
  columns; selected bar figures render those CI values.
- Fig. 14 now uses a direct SGLSim DP-size sweep instead of the r5-derived
  scale projection: DP=2 uses the r5 matrix, and DP=4/DP=8 add 3 seeds each.
  The sweep completed 1,320/1,320 rows with 264 hard-failure cross-rank replay
  rows. Normalized duration is 1.000 at DP=2, 0.985 at DP=4, and 0.983 at
  DP=8. This is a serving-path scale smoke, not final high-concurrency scale
  evidence.
- Fig. 14b now uses an implemented DP=4 online peer-route sweep with explicit
  retry route policies. The non-peer, random-healthy, and AegisKV peer-route
  scenarios share the same short-tail workload and seeds `8500/8501/8502`.
  AegisKV peer routing hits peer GPUReplica on 72/72 replay rows, while random
  healthy routing hits 21/72 and non-peer routing hits 0/72. AegisKV peer-route
  WCT is 0.988x the non-peer route and p99 recovery overhead is 74.3% of the
  non-peer recompute route.
- Agentic online suites `sglsim-agentic-online-r3-20260621`,
  `sglsim-agentic-peer-hot-tail-r3-20260621`, and
  `sglsim-physical-copy-agentic-r3-20260628` are retained as provenance. The
  current Fig. 10/Fig. 14 online rows are superseded by
  `sglsim-physical-recovery-action-agentic-r3-20260628`, which repeats the
  hard-failover scenarios with recovery-shaped Host/GPUReplica movement curves
  driving restore timing.
- Final HPCA claims still require replacing those preliminary CSVs with
  `evidence_tier=final` SGLang/SGLSim measurements.

## Fig. 1：同等大小 KV loss 动机

Question:

```text
Does equal-size KV loss have non-uniform WCT impact?
```

Plot:

- x-axis: segment position / lifecycle。
- y-axis: delta workflow completion time。
- bars: terminated branch, running subagent, last subagent, main resume, synthesis tail, shared prefix。

Expected result:

- terminated branch near zero；
- final/synthesis/main resume much larger；
- shared prefix causes replay amplification。

What it proves:

> AegisKV needs workflow-aware value; byte size alone is insufficient.

## Fig. 2：Restore 与 recompute 边界

Question:

```text
When should a retry restore KV and when should it recompute?
```

Plot:

- x-axis: segment tokens or KV bytes。
- y-axis: recovery time。
- lines: GPUReplica restore, Host restore, recompute。

Expected result:

- recompute wins for tiny suffixes when peer GPUReplica is unavailable or not
  worth placing；
- GPUReplica wins for small high-value hot tails if peer is not congested；
- Host wins for long prefixes when H2D is cheaper than replay；
- AegisKV tracks the lower envelope。

Current evidence:

- `fig_restore_recompute_boundary.csv` is now calibrated from direct SGLSim
  hard-failure replay rows, run id
  `sglsim-fig2-restore-boundary-calibrated-r11-20260621`.
- Inputs are the same three regimes used by Figs. 9-10: 120 Host rows,
  72 GPUReplica rows, and 72 recompute rows, for 264 replay rows total.
- Fitted runtime coefficients are `recompute = 4.0 + 0.0108 * tokens`,
  `Host = 7.5 + 0.0036 * tokens`, and `GPUReplica = 2.4 + 0.0019 *
  peer_tokens + 0.0108 * suffix_tokens`; observed peer-token coverage in the
  GPUReplica regime is 34.2%.
- This is SGLSim runtime replay-calibrated evidence with repeated SGLang
  raw H2D/P2P copy calibration from jobs `34338/34344/34345` and
  recovery-shaped Host/GPUReplica restore movement from jobs
  `34357/34358/34359` overlaid.
  Final Fig. 2 still needs the physical copy paths inside the end-to-end
  recovery flow.

What it proves:

> Backup hit is not enough; recovery action must be restore/recompute-aware.

## Fig. 3：静态 GPUReplica budget 差距

Question:

```text
Is one fixed GPUReplica budget enough on top of fixed HostBackup?
```

Plot:

- x-axis: GPUReplica budget fraction。
- y-axis: WCT or expected recovery cost。
- lines: different workloads/failure timings。

Expected result:

- hot-tail rows improve as peer-visible tails are covered, then saturate；
- long-prefix rows prefer Host restore over forced peer splitting；
- fan-out/non-peer rows do not improve from HBM budget without retry-rank
  visibility；
- dynamic AegisKV remains a single online choice rather than one static ratio。

Current preliminary input:

- `fig_static_gpu_replica_budget_gap.csv` now comes from implemented online
  SGLSim DP4 hard-restart budget scenarios, run id
  `sglsim-budget-sweep-agentic-r1-combined-20260628`；
- rows cover hot-tail, long-prefix, and fan-out/non-peer workloads at
  `0/0.1/0.2/0.5/1.0` GPUReplica budget fractions；
- the sweep completed 600 replay rows across Fig. 3 and Fig. 13 inputs with
  0 failed rows; Fig. 3 directly uses 360 static-curve replay rows plus a
  dynamic AegisKV summary row；
- this is implemented online policy evidence, but still metadata
  materialization rather than final physical KV-copy recovery。

What it proves:

> AegisKV is an online resource partitioning problem, not a static rule.

## Fig. 4：系统总览

Content:

```text
Workflow metadata -> SGLang request -> radix node Aegis metadata
-> placement planner -> GPUReplica / Host -> safe frontier
-> failure retry -> restore/recompute
```

What it proves:

> AegisKV is integrated with SGLang's KV lifecycle, not a separate KV manager.

## Fig. 5：备份数据路径

Content:

- source DP rank active KV；
- peer DP rank GPUReplica；
- same worker Host memory；
- retry rank；
- P2P / D2H / H2D arrows；
- ACK path。

What it proves:

> GPUReplica and Host have different latency, capacity, and fault-domain semantics.

## Fig. 5b：固定 HostBackup visibility pipeline

Content:

```text
GPU active KV
  -> radix insert
  -> fixed write_through D2H
  -> Host ACK
  -> HostBackup directory entry
  -> source GPU failure
  -> retry on another DP rank
  -> lookup failed-rank Host KV
  -> Host restore / suffix recompute

GPU active KV
  -> peer-copy queue
  -> peer GPUReplica ACK
```

Include queue boxes:

- D2H write queue；
- H2D load/prefetch queue；
- HostBackup directory；
- peer-copy queue；
- ACK queues。

Expected result if plotted with measurements:

- baseline and AegisKV pay the same fixed write-through D2H；
- baseline cannot recover because retry rank lacks failed-rank Host frontier visibility；
- AegisKV recovers ACKed Host prefixes and recomputes only missing suffixes。

What it proves:

> AegisKV's Host contribution is cross-rank recovery visibility over an existing fixed write-through Host cache, not a new D2H strategy.

## Fig. 6：在线决策循环

Borrowed structure from One Pool:

```text
observe state -> choose budget/placement -> physical copy -> ACK
-> update frontier -> failure-time restore -> observe next state
```

Include state vector table:

- workflow state；
- KV state；
- backup state；
- resource pressure；
- failure exposure；
- restore/recompute profile。

## Fig. 7：故障恢复时间线

Timeline:

```text
normal decode
backup copy
ACK
failure
snapshot
retry admission
frontier lookup
restore/recompute
suffix replay
resume workflow
```

Expected comparison:

- no backup has longest suffix replay；
- Host reduces prefix replay but pays H2D；
- GPUReplica fastest when it hits；
- AegisKV selects the best path by segment。

## Fig. 7b：Radix lock、ACK 与 eviction 状态机

Content:

```text
device-resident unlocked
  -> active request lock_ref > 0
  -> Host copy pending
  -> Host ACKed safe
  -> GPU evicted / Host-only
  -> H2D restored
  -> Host evicted / unsafe
```

Annotate:

- `inc_lock_ref` / `dec_lock_ref`；
- `writing_check` ACK；
- `evict` and `evict_host`；
- safe frontier valid/invalid transitions。

Expected result:

- pending copy is never marked safe；
- locked nodes are not evicted；
- Host-only frontier is recoverable only after ACK；
- Host eviction invalidates frontier。

What it proves:

> The system design respects SGLang radix correctness rather than treating KV backup as an external blob copy.

## Fig. 8：整体 workflow completion time

Policies:

```text
no backup
uniform Host
uniform GPUReplica
AegisKV
```

Expected result:

- AegisKV best or near-best across skewed workflows；
- uniform GPUReplica may be best for tiny high-value tails but poor under HBM pressure；
- uniform Host stable for long prefixes but can be slower on short segments；
- no backup is the same-row recompute baseline。

Current evidence:

- `fig_overall_wct.csv` now comes from an implemented online SGLSim policy
  sweep rather than a replay counterfactual. The active run id is
  `sglsim-policy-sweep-agentic-r6-dp4-matched-seed-20260628`.
- The sweep runs four policies over the same DP4, 32-agent, 3-turn workload
  with matched seeds `8100/8101/8102`. Each policy contributes 360 completed
  rows, 72 hard-failure replay rows, and 0 failed rows.
- Normalized mean end-to-end WCT is 1.000 for no backup, 0.982 for uniform
  Host, 0.982 for uniform GPUReplica, and 0.976 for AegisKV. The low
  end-to-end improvement is expected because this low-concurrency smoke
  workload is dominated by normal serving time rather than restore overhead.
- This is online SGLSim serving-path evidence with explicit
  `gsp_aegis_policy`, but it remains metadata-level. Final Fig. 8 still needs
  live physical tensor movement or paired real-system validation.

## Fig. 9：恢复延迟与 retry speedup

Metrics:

- critical-agent recovery latency；
- retry latency；
- speedup over no backup。

Expected result:

- GPUReplica has lowest hit latency；
- Host has moderate latency but better coverage；
- AegisKV combines them and improves tail。

Current evidence:

- `fig_recovery_latency.csv` is derived from the implemented online policy
  sweep `sglsim-policy-sweep-agentic-r6-dp4-matched-seed-20260628`.
- Each plotted policy uses the same DP4, 32-agent, 3-turn workload and matched
  seeds `8100/8101/8102`; each policy contributes 72 hard-restart replay rows
  and 0 failed rows.
- AegisKV is 0.780x no-backup recompute for mean retry latency and 0.915x on
  critical-agent p90 latency. Uniform Host is 0.793x/0.908x and uniform
  GPUReplica is 0.829x/0.935x.
- This is online SGLSim serving-path latency evidence with metadata-level
  restore materialization; strict-final Fig. 9 still needs physical CUDA KV-copy
  validation or paired real-system recovery validation.

## Fig. 10：恢复路径拆解

Stacked bar:

```text
peer_gpu
host_h2d
ordinary_cache
recompute
```

Expected result:

- AegisKV uses peer GPU selectively for high-value short segments；
- Host covers long prefix；
- recompute remains for low-value or unsafe-frontier segments。

Current evidence:

- `fig_recovery_path_breakdown.csv` is derived from the multi-regime replay
  counterfactual retained for Figs. 9-10. The active run id is
  `sglsim-fig8-10-policy-counterfactual-r11-20260621`.
- Across 264 replay rows and 230,327 runtime recovery tokens, AegisKV's
  equal-regime mix is 9.7% peer GPU, 33.3% Host H2D, 0.0% ordinary cache, and
  56.9% recompute. Uniform Host is 100% Host H2D; uniform GPU is 14.0% peer GPU
  and 86.0% suffix recompute under the same regime mix.
- `sglsim-physical-recovery-action-agentic-r3-20260628` appends four repeated online
  action-distribution rows with recovery-action-calibrated restore timing: DP2
  Host, DP4 peer, DP4 non-peer, and DP4 peer hot-tail. The first two
  long-prefix scenarios split mostly to Host restore with small peer-GPU
  hot-tail shares; DP4 non-peer chooses Host restore; DP4 peer hot-tail observes
  72/72 `gpu_replica` restore actions and yields 19.2% peer-GPU restored tokens
  plus 80.8% suffix recompute.
- This now exercises Host, peer, and recompute lower-envelope choices for all
  replay-calibrated policies, while separately documenting the current online
  agentic-suite action coverage. Jobs `34338/34344/34345` add repeated raw
  H2D/P2P/D2H copy calibration, and jobs `34357/34358/34359` add the
  recovery-shaped Host/GPUReplica restore annotation used by the plot. Final
  Fig. 10 still needs physical CUDA KV-copy inside end-to-end recovery policy
  runs before being relabeled as final measured evidence.

## Fig. 10b：Shared-prefix marginal accounting

Question:

```text
How much duplicate backup does AegisKV avoid when agents share prefixes?
```

Plot:

- x-axis: shared prefix ratio。
- y-axis left: duplicated protected bytes or write amplification。
- y-axis right: recovery coverage。
- bars/lines: no shared-prefix dedup vs marginal SafeCover accounting。

Expected result:

- no-dedup repeatedly protects each turn's safe frontier；
- marginal accounting protects only newly extended frontier tokens for the same
  agent；
- reconstruction coverage stays at the final frontier；
- benefit grows with observed shared-prefix overlap。

Current preliminary input:

- original request rows from the three SGLSim hard-failure matrices used by
  Figs. 2, 3, 8, 9, 10, and 12；
- retry rows are excluded to avoid using retry cache-hit tokens as the
  shared-prefix signal；
- current x-axis covers observed workload bins, not a full 0-95% synthetic
  sweep。

What it proves:

> AegisKV's two backup caches are coupled by shared KV content, unlike independent One Pool caches.

## Fig. 11：Normal-path overhead

Metrics:

- p95/p99 no-failure request latency；
- throughput；
- P2P copy BW；
- D2H/H2D BW；
- backup write amplification。

Expected result:

- AegisKV overhead bounded；
- uniform backup creates higher overhead；
- no backup has zero overhead but poor recovery。

Current evidence:

- `hkbugpumanode` jobs `32861`, `32863`, `32869`, `32870`, `32871`, `32874`,
  and `32899` are matched SGLang DP=2 overhead smokes. Together they show
  HiCache write-through mean overhead average 0.22%, p95 overhead average
  -0.19%, and p99 overhead average 0.31% over a no-HiCache baseline under
  matched CUDA
  graph controls. The per-run range crosses zero, so this is a "within noise"
  overhead anchor, not a precise positive tail-overhead claim.
- The current Fig. 11 measured CSV is therefore `mixed-preliminary`: SGLSim
  r3 failover-matrix policy-candidate rows are shown next to the SGLang paired
  aggregate rows so the paper layout exposes both the scale model and the
  real-system overhead anchor.

## Fig. 11b：固定 write-through 下的 HostBackup visibility

Question:

```text
If both systems already perform write_through D2H, what does AegisKV add?
```

Plot:

- grouped bars for `baseline_write_through_invisible`, `host_visible_no_gpu_replica`, `gpu_replica_only`, `AegisKV`。
- y-axis 1: no-failure p99 overhead relative to fixed-write-through baseline。
- y-axis 2: retry latency, recovered tokens, or WCT under failure。

Expected result:

- baseline pays D2H but cannot use failed-rank Host KV after retry；
- Host-visible recovery reduces recomputation for long prefixes；
- GPUReplica-only is fast when it hits but limited by peer HBM/locality；
- AegisKV combines Host visibility with selective GPUReplica。

Current preliminary input:

- retry-latency axis uses direct SGLSim hard-failure replay ratios against
  same-row no-backup recompute；
- Host visible uses long-prefix Host-visible rows, GPUReplica-only uses
  peer-visible hot-tail rows, and AegisKV aggregates both recoverable regimes；
- overhead axis is still a smoke/model anchor and needs paired final policy
  measurements。
- SGLang support artifact `failure_visibility_pair_32978_32980` compares job
  `32978` without a storage backend against job `32980` with `remote_backup`.
  It validates a 5,132-token Host frontier delta but also records 1.108x p99
  latency for the storage-backed smoke, so it should be cited as frontier
  visibility evidence rather than a recovery-latency win.
- Jobs `32981` and `32982` identify the next implementation boundary:
  split-prefix prompts can leave backed leaf data at `page_start=14`, causing a
  contiguous-prefix storage miss from a retry rank with no local prefix; a
  single-request page-0 shape reaches storage match/fetch. Job `32983` shows
  that `wait_complete` materializes those accepted pages before prefill
  admission (`#cached-token: 1297`). Job `32985` confirms that `wait_complete`
  alone does not fix the shared-prefix four-request case, job `32987` confirms
  that four-request materialization works when shared-prefix radix splitting is
  removed, and job `32991` confirms the patched split-prefix storage path
  materializes four shared-prefix retries with `#cached-token: 1297`. Jobs
  `32998`/`32999` add the fixed-write-cost visibility control: both runs
  perform the same remote-backup writes, but only the visible run can prefetch
  the Host frontier during retry, materializing `#cached-token: 5137`. Jobs
  `33002`/`33003` add the same control under `best_effort`, where the visible
  run improves p99 but only partially materializes the Host frontier.

What it proves:

> The HostBackup benefit comes from failure-aware metadata and restore protocol, not from changing the normal D2H policy.

## Fig. 12：Ablation

Ablations:

- no lifecycle；
- no segment granularity；
- no restore/recompute；
- no resource pressure；
- no hysteresis；
- no critical restore scheduling。

Expected result:

- no lifecycle and no critical restore regress because retry cannot use the
  recoverable frontier and falls back to recompute；
- no segment granularity and no pressure regress in the hot-tail workload
  because they suppress the peer GPUReplica path；
- no restore/recompute regresses because it forces Host restore even when peer
  GPUReplica is the lower-envelope action；
- no hysteresis regresses by intermittently losing a stable frontier across
  turns。

Current preliminary input:

- `sglsim-ablation-sweep-agentic-r4-dp4-hot-tail-matched-seed-20260628`；
- 7 implemented online ablation toggles, 3 matched seeds each, DP4, 32 agents,
  3 turns, 21 trials, 0 failed rows, 504 replay rows；
- AegisKV is normalized to `1.0`; every disabled variant is above the full
  policy in observed WCT；
- strict-final still requires live physical KV movement rather than SGLSim
  metadata materialization with calibrated timing。

## Fig. 13：Budget sensitivity heatmap

Axes:

```text
x = GPUReplica budget
y = failure timing
color = WCT or expected recovery cost
```

Expected result:

- per-regime optimum shifts with failure timing and overlap；
- more backup is not always better；
- high GPUReplica budget can hurt normal path。

Current preliminary input:

- `fig_budget_sensitivity.csv` now comes from the same implemented online
  SGLSim DP4 hard-restart sweep, run id
  `sglsim-budget-sweep-agentic-r1-combined-20260628`；
- rows cover early, mid, and late decode failure timing with
  `0/0.1/0.2/0.5/1.0` GPUReplica budget fractions；
- the heatmap directly aggregates 360 replay rows with 0 failed rows；
- this is hybrid-ready online sweep evidence, but strict-final still needs
  physical KV movement instead of metadata materialization。

## Fig. 14：DP scale 与 peer availability

Axes:

- DP size: 2, 4, 8；
- workflow count / concurrency；
- WCT or retry latency。

Expected result:

- the current preliminary direct sweep should complete at DP=2, DP=4, and DP=8
  without failed requests；
- duration should remain stable under low-concurrency hard-failover replay；
- final evidence must add high-concurrency workflow-count sweeps。

## Fig. 14b：Peer-rank selection

Question:

```text
How should GPUReplica choose the peer DP rank?
```

Plot:

- x-axis: retry route choice。
- bars: non-peer route miss, random healthy route, recorded AegisKV peer route。
- y-axis: normalized WCT under hard-failover replay。
- secondary metric: p99 recovery overhead as a percentage of the non-peer
  recompute route。

Expected result:

- non-peer route misses peer GPUReplica and falls back to recompute；
- random healthy routing recovers only with probability `1/(dp_size-1)`；
- retry-local peer routing gets the lower recovery overhead when peer tokens
  are visible。

Current preliminary input:

- run id: `sglsim-peer-route-sweep-agentic-r1-dp4-matched-seed-20260628`；
- scenarios: `peer_route_nonpeer_hot_tail`, `peer_route_random_hot_tail`, and
  `peer_route_aegiskv_hot_tail`；
- each scenario contributes 360 completed rows, 72 hard-failure replay rows,
  and 0 failed rows；
- route hit rates: non-peer 0/72, random healthy 21/72, AegisKV peer 72/72；
- current AegisKV peer route: 0.988x normalized WCT and 74.3% p99 recovery
  overhead relative to the non-peer recompute route。

What it proves:

> GPUReplica is not just a capacity allocation problem; retry-rank locality
> determines whether the protected peer tokens are usable.

## Fig. 15：SGLang/SGLSim calibration anchor

Plot:

- x-axis: SGLang p99 overhead anchor；
- y-axis: SGLSim p99 overhead anchor；
- labels: baseline, protected path。

Expected result:

- current anchor exposes whether SGLSim is optimistic or conservative on
  normal-path overhead；
- the 32978/32980 failure-visibility pair contributes a backed-up-token
  frontier calibration point, but not a ranking-preservation point；
- job 32983 contributes a single-request materialized restore calibration
  point after job 32982 isolated the storage match/fetch path；
- job 32985 contributes a four-request negative calibration point for the
  original split-prefix restore boundary；
- job 32987/32989 contributes a four-request unique-prefix materialized
  restore calibration pair；
- job 32991 and the 32992/32991 pair contribute a four-request shared-prefix
  materialized restore calibration pair after the split-prefix storage fix；
- job 32993/32994 extends Host-frontier calibration to a longer shared prefix,
  but the visible path is still slower than control；
- job 32998/32999 contributes a same-write-cost visibility isolation pair:
  both sides write 20,548 tokens, but only the visible side can consume the
  Host frontier during retry；
- job 33002/33003 contributes a same-write-cost `best_effort` latency point:
  partial Host restore improves p99 to 0.966x but does not prove full restore；
- final paired policy calibration should preserve relative ranking；
- exact latency equality is not required。

Prerequisite:

```text
SGLSim must add final visible AegisKV policy hooks and paired online workflow
runs before this figure can become a full policy-ranking calibration. Current
data only anchor normal-path overhead.
```

## Table 1：状态向量

Mirror One Pool's state-vector table, but for AegisKV:

| Symbol | Meaning |
|---|---|
| `h_t` | Host backup hit / ACK ratio |
| `g_t` | GPUReplica ACK ratio |
| `q_t` | retry / scheduler queue pressure |
| `p2p_t` | P2P bandwidth pressure |
| `h2d_t` | H2D restore pressure |
| `d2h_wt_t` | fixed write-through D2H pressure |
| `host_vis_t` | fraction of ACKed Host KV visible to retry ranks |
| `w_t` | workflow criticality skew |
| `l_t` | lifecycle distribution |
| `loc_{p,t}` | peer KV locality for candidate rank `p` |
| `load_{p,t}` | peer queue/load pressure |
| `dup_t` | duplicate Host/GPUReplica protected bytes |
| `alpha_gpu_t` | peer GPU backup budget ratio |
| `r_t` | recent recovery success / saved WCT |

## Table 2：Baseline

Show which signals each policy uses:

| Policy | Workflow | Lifecycle | Segment | Restore/recompute | Resource pressure |
|---|---|---|---|---|---|
| no backup | no | no | no | recompute | no |
| baseline write-through invisible | no | no | no | recompute | fixed D2H only |
| Host visible only | no | no | segment | restore/recompute | Host restore queue |
| uniform GPUReplica | no | no | segment | restore | peer budget only |
| request-level | weak | no | no | restore | coarse |
| agent-level | yes | weak | no | restore | coarse |
| static critical path | yes | weak | segment | restore | no |
| AegisKV | yes | yes | yes | yes | yes |

## Table 3：实现 hook

| Hook | SGLang path | Purpose |
|---|---|---|
| request metadata | tokenizer manager / `Req` | attach workflow state |
| radix match/insert/split | `HiRadixCache` / `RadixCache` | update segment metadata and shared SafeCover |
| fixed Host D2H | `HiRadixCache.write_backup` / `HiCacheController.write` | observe fixed write-through copy |
| Host ACK | `HiRadixCache.writing_check` / `ack_write_queue` | publish HostBackup directory entry |
| H2D/cross-rank restore | `HiCacheController.load` / restore protocol | recover failed-rank Host frontier on retry rank |
| GPU eviction | `HiRadixCache.evict` | preserve/invalidate GPU hit state |
| Host eviction | `evict_host` | invalidate Host frontier |
| peer ACK | peer copy manager / DP controller | mark GPUReplica frontier |
| peer demotion | peer replica manager | invalidate or downgrade GPUReplica frontier |
| failure retry | scheduler / DP controller | choose frontier and retry path |

## Table 4：正确性测试

| Test | Expected invariant | Current artifact |
|---|---|---|
| missing/pending HostBackup + failure | pending frontier rejected; retry recomputes | `test_aegis_restore_lookup_recomputes_when_host_backup_missing` |
| active `lock_ref` + eviction pressure | locked restored prefix is not evicted | `test_aegis_restored_locked_prefix_is_not_evicted_under_pressure` |
| divergent radix insert after restore | restored frontier token span still matches | `test_aegis_restored_prefix_survives_divergent_radix_insert` |
| unsupported pending restore action | no device KV is materialized before ACK | `test_aegis_pending_restore_action_does_not_materialize_device_cache` |
| Host unavailable + peer unavailable | unsafe frontier is not selected | `test_aegis_restore_lookup_recomputes_when_host_evicted_and_peer_unavailable` |
| peer demotion | GPUReplica frontier used only on peer rank | `test_aegis_restore_lookup_demotes_peer_gpu_when_retry_rank_is_not_peer` |
| request without Aegis metadata | no Aegis frontier or candidate retry is published | `test_aegis_baseline_request_without_metadata_does_not_publish_frontier` |

## 最小 paper-ready 结果包

The first complete paper should contain:

1. Fig. 1 motivation。
2. Fig. 2 restore/recompute boundary。
3. Fig. 4 system overview。
4. Fig. 7 failure timeline。
5. Fig. 8 overall WCT。
6. Fig. 10 recovery path breakdown。
7. Fig. 11 normal-path overhead。
8. Fig. 12 ablation。
9. Fig. 15 SGLang/SGLSim calibration。
