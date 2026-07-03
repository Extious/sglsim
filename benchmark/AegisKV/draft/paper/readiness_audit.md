# AegisKV HPCA 2027 完成度审计

本文用于审计当前草稿和已生成图表是否接近一篇完整的 HPCA 风格系统论文。审计标准刻意偏严格：expected-trend template 和 preliminary counterfactual plot 可以帮助打磨叙事和版式，但它们不是最终实验证据。

## 当前状态

当前草稿已经具备完整图表包：

- 25 张 PNG 渲染图：1 张生成式 System overview，以及 24 张 motivation / experiment 定量图。
- 旧的详细 draw.io 机制流程图已经从当前图像包中移除；Beamer 只保留生成式 System overview 和测得/派生图像。
- 24 个带 CSV 输入的定量结果图，其中 23 个当前是 preliminary / hybrid-ready。
- 1 个 strict-final-ready SGLSim measured CSV：
  - Fig. 10b under
  `figures/data/measured/fig_shared_prefix_marginal.csv`: run
  `codex-shared-prefix-sweep-r3-20260628` covers nine online SGLSim trials,
  480 request rows, 141 multi-turn agent groups, and shared-prefix bins from
  2.5% to 77.5%.
- Fig. 1/1b/1c、Fig. 3b/3c/3d 等动机图当前是 trace-derived 或 model/online
  hybrid-ready 证据，不再声称为 strict-final。Fig. 3b 的 MDP policy table 已由
  runtime `choose_alpha` 逐项复现；Fig. 3c 的 270 条 raw request trace 逐行重放
  均与 runtime alpha 记录一致。
- `figures/data/measured/` 下还有 SGLSim-derived、mixed preliminary、
  SGLang-paired anchor、physical-copy calibration 和 base SGLang/SGLSim accuracy
  CSV。它们报告 run-level 或 request-level std/95% CI 列，仍 kept below
  `evidence_tier=final`，除非 final gate 明确接受。
- Tables 1-4 已保存为 Markdown artifact。
- manifest 会将生成式 System overview 标为 `design-ready`，并在存在 preliminary CSV 时将结果图标为 `sglsim-preliminary`。
- validator 会写出 `figures/evidence_status.json`；在所有结果图都有 `evidence_tier=final` measured CSV 之前，`--strict-final` 会失败。当前 strict-final-ready 定量图只有 Fig. 10b；HPCA hybrid gate 为 25/25 ready。
- draft integrity checker 会检查章节链接、figure 输出、preliminary/final CSV 是否存在，以及 figure evidence gate。
- `experiments/final_evidence_matrix.csv` 中保存了 per-figure final evidence matrix，
  mapping each quantitative result figure to its required SGLang/SGLSim
  replacement evidence.
- The problem formulation has been rebuilt in an HLEM / One Pool style:
  resource pool, state vector, budget action, objective, value-density online
  policy, constraints, and failure-time recovery rule are now written with
  renderable LaTeX equations. The model explicitly treats HostBackup as a
  fixed SGLang `write_through` state and peer GPUReplica as the active
  placement decision.
- `current_results_overview.md` now explains the current SGLSim request
  construction path, multi-turn metadata, routing-key affinity, hard-failover
  replay injection, runtime Host/GPUReplica directory lookup, restore sleep,
  and metadata-level materialization boundary.
- SGLang small-scale Slurm scaffolding under `experiments/sglang_small/`.
- SGLang DP=2 smoke evidence on `hkbugpumanode`: job `32855` completed with
  8/8 `/v1/completions` requests on Qwen3-8B, DP=2, HiCache write-through.
- SGLang paired no-failure overhead smoke on `hkbugpumanode`: job `32861`
  completed baseline and HiCache write-through cases on the same A100 node
  with matched CUDA graph controls; both cases completed 8/8 requests and
  HiCache write-through measured 2.02% p99 overhead over baseline.
- `sglang_small/scripts/aggregate_overhead_pairs.py` now aggregates matched
  paired overhead runs and excludes confounded runs. It includes jobs `32861`,
  `32863`, `32869`, `32870`, `32871`, the 64-request job `32874`, and the
  128-request job `32899`, and excludes job `32859` because its baseline did
  not disable piecewise CUDA graph behavior. The current matched aggregate
  shows mean overhead average 0.22% with range -2.23% to 3.05%; p95 overhead
  average -0.19% with range -4.09% to 2.02%; and p99 overhead average 0.31%
  with range -3.93% to 3.74%,
  so the tail-latency claim should be "within noise" rather than a precise
  positive overhead number.
- SGLang DP=2 failure-control-path smokes on `hkbugpumanode`: jobs `32977` and
  `32978` both completed 4/4 routed in-flight Qwen3-8B completion requests
  after `/simulate_gpu_failure` on DP rank 0. The real SGLang event logs
  captured and re-dispatched all 4 active requests, emitted
  `gpu_failure_simulated`, `failover_dispatched`, `resume_accepted`, and
  `gpu_recovered` events, and all final responses carried `sglext.failover`.
  Job `32978` is the stronger smoke: failure arrived after 190 generated
  tokens per request, 760 pre-failover output tokens total. Both smokes still
  reported 0 pre-failover backed-up tokens, so this is retry continuity
  provenance, not final Host-visible restore or GPUReplica physical KV-copy
  evidence.
- SGLang DP=2 storage-backed failure smoke on `hkbugpumanode`: job `32980`
  launched a centralized `remote_backup` server and SGLang workers with
  `hicache_storage_backend=remote_backup`. It completed 4/4 routed in-flight
  requests after `/simulate_gpu_failure`, returned 4 `sglext.failover`
  responses, and reported 5,132 pre-failover backed-up tokens across the 4
  requests. This is the first real SGLang smoke in the current artifact set
  that validates a nonzero Host-visible recoverable frontier. It is still not
  final recovery-latency evidence because it is a four-request control-path
  smoke with overlap disabled, not a final matched Host-visible/Host-invisible
  restore-policy comparison.
- SGLang 32978/32980 failure-visibility pair: the posthoc artifact
  `results/failure_visibility_pair_32978_32980/` compares the no-storage
  control smoke against the `remote_backup` smoke. It records a Host frontier
  delta of 5,132 backed-up tokens, 1,283 backed-up tokens/request, and 1.108x
  p99 latency for the storage-backed smoke relative to no-storage. This is the
  first paired Fig. 11b/Fig. 15 SGLang frontier artifact, but it remains a
  frontier-visibility smoke rather than a final recovery-benefit measurement.
- SGLang 32981-32991 restore-path probes: job `32981` disables request-aware
  remote-backup retention and still gets `data_pages=0` on retry, showing that
  lease GC is not the main miss cause. The miss is instead tied to split radix
  prefixes: the four-request shape writes backed leaf data from `page_start=14`
  while the retry rank queries from page 0. Job `32982` removes that shape with
  a single request and reaches `data_pages=1297`, `Prefetch ACCEPTED`, and
  `prefetch_get` hits, but the default `best_effort` policy terminates before
  IO completion and reports `completed=0`.
- SGLang 32983 wait-complete restore probe: job `32983` keeps the same
  single-request page-0 remote-backup shape as job `32982` and switches
  `hicache_storage_prefetch_policy` to `wait_complete`. It completes prefetch
  IO with `completed=1297 / expected=1297` and retry prefill admits with
  `#cached-token: 1297`. This resolves the basic materialization gap for the
  small SGLang restore-path smoke. It is still not final HPCA evidence because
  it is not a matched Host-visible/Host-invisible policy comparison under
  representative concurrency and normal-path cost.
- SGLang 32984/32983 single-request pair: job `32984` is the no-storage
  single-request control for job `32983`. The paired artifact
  `results/failure_visibility_pair_32984_32983/` records a 1,297-token Host
  frontier delta and 1.124x visible/no-storage p99 latency. This is a
  correctness/calibration pair, not a latency-benefit result.
- SGLang 32985 four-request wait-complete probe: job `32985` keeps the
  storage-backed four-request shape and switches to `wait_complete`. It still
  completes 4/4 failover responses and reports 5,132 backed-up tokens, but
  retry-side prefetch sees `data_pages=0` for all four requests. This confirms
  that the current concurrent shape is blocked by split-prefix/storage-layout
  matching, not by the prefetch stop policy fixed in job `32983`.
- SGLang 32987/32989 unique-prefix four-request pair: job `32987` removes the
  shared-prefix split by setting `UNIQUE_REQUEST_PREFIX=1`, keeps
  `remote_backup + wait_complete`, and restores four concurrent failover
  requests. All four retry queries report `data_pages=1301`; all four prefetch
  IOs complete `1301 / 1301`; all four retry prefills admit with
  `#cached-token: 1301`. Job `32989` supplies the no-storage unique-prefix
  control. The paired artifact `results/failure_visibility_pair_32989_32987/`
  records a 5,204-token Host frontier delta and 1.308x visible/no-storage p99
  latency. This proves concurrent materialized restore is possible in real
  SGLang when backed data starts at page 0.
- SGLang 32992/32991 shared-prefix four-request pair: job `32991` applies the
  split-prefix storage fix, keeps the original shared-prefix four-request
  workload, and restores all four failover retries with `data_pages=1297`,
  `completed=1297 / expected=1297`, and retry `#cached-token: 1297`. The paired
  artifact `results/failure_visibility_pair_32992_32991/` compares this
  patched storage-backed run against the no-storage shared-prefix control job
  `32992`, recording a 5,188-token Host frontier delta and 1.210x
  visible/no-storage p99 latency. This supersedes the unique-prefix pair as the
  strongest current SGLang correctness/calibration artifact for shared-prefix
  restore, but it is still not final recovery-benefit evidence.
- SGLang 32993/32994 long shared-prefix pair: job `32993` is a no-storage
  control with `PROMPT_REPEAT=256` and `FAILURE_DELAY_S=12`; job `32994` is the
  matched `remote_backup + wait_complete` run. The pair completes 4/4 failover
  responses on both sides. The visible run reports 20,548 backed-up tokens
  versus 0 for the control, and all four visible retries materialize
  `data_pages=5137` with retry `#cached-token: 5137`, but p99 latency is
  35.96 s versus 22.06 s (`1.630x`). This extends Host-frontier calibration to
  a larger prefix, but reinforces that the current remote-backup smoke path is
  correctness evidence, not final recovery-benefit evidence.
- SGLang 32998/32999 same-write-cost visibility pair: both jobs enable
  `remote_backup + wait_complete`, `PROMPT_REPEAT=256`, and `FAILURE_DELAY_S=12`
  and both report 20,548 backed-up tokens. Job `32998` disables failover-time
  storage prefetch, so it pays the same remote-backup write path while hiding
  HostBackup from retry; job `32999` enables lookup and materializes
  `data_pages=5137` plus retry `#cached-token: 5137` for all four requests.
  The visible p99 latency is 39.31 s versus 26.42 s (`1.487x`). This is the
  cleanest fixed-normal-write-cost Fig. 11b SGLang isolation artifact, but the
  wait-complete restore path still does not demonstrate a latency benefit.
- SGLang same-write-cost `best_effort` visibility repeats: jobs `33002/33003`,
  `34333/34334`, and `34336/34337` all enable `remote_backup` writes on both
  sides while only the visible job enables failover-time lookup. The visible
  retries find `data_pages=5137` and partially materialize before retry prefill
  (`#cached-token: 2830` in `33003`, `#cached-token: 2318` in `34334`,
  `#cached-token: 1806` in `34337`). The three repeats aggregate to 0.961x
  visible/control p99 with 0.016 95% CI, and all three repeats improve p99.
  This is now the cleanest same-write-cost SGLang
  latency-improvement artifact, while 32998/32999 remains the clean full-restore
  materialization artifact.
- SGLang physical Host-restore parser: `sglang_physical_restore_evidence_summary.md`
  and `results/physical_restore_evidence/physical_restore_runs.csv` now extract
  `data_pages`, accepted prefetch pages, completed prefetch tokens, and retry
  `#cached-token` from real SGLang `server.log` files. The structured rows
  separate hidden-frontier controls, full wait-complete materialization, and
  partial best-effort asynchronous materialization.
- SGLang physical CUDA copy microbench: jobs `34338`, `34344`, and `34345`
  run on `hkbugpusrv14` with two NV12-connected A100 GPUs and measure repeated
  KV-sized tensor movement under a 147,456 bytes/token Qwen3-8B bf16
  assumption. At the 8192-token point, aggregate p50 latency is 48.595 ms for
  H2D (95% CI 0.531 ms), 46.416 ms for D2H (95% CI 0.100 ms), and 4.368 ms for
  GPU0-to-GPU1 P2P (95% CI 0.004 ms). This is physical copy calibration for
  Fig. 2/Fig. 10, not end-to-end AegisKV recovery policy evidence.
- SGLang physical recovery-path microbench: jobs `34357`, `34358`, and `34359`
  run the same KV-size assumption but shape movement as AegisKV backup/restore
  actions that write into retry-rank KV buffers. At 8192 tokens, Host restore
  H2D is 48.434 ms p50 (95% CI 0.013 ms), GPUReplica restore P2P is 4.330 ms
  p50 (95% CI 0.005 ms), Host backup write D2H is 46.521 ms p50, and
  GPUReplica backup write P2P is 4.324 ms p50. This strengthens Fig. 2/Fig. 10
  physical movement evidence, but still does not replace live end-to-end policy
  recovery.
- SGLSim local smoke evidence: `sglsim_large` launches
  `benchmark/synthesis_requests/run.py` with `mock-model` and completed 8/8
  generated-shared-prefix turn requests.
- SGLSim serving-derived policy-candidate evidence: `sglsim_large` now has
  `scripts/derive_policy_candidates_from_synthesis.py` and
  `scripts/run_policy_candidate_smoke.sh`. The first larger candidate run
  completed 96/96 request rows with 0 failures and emitted schema-valid
  candidate CSVs for all 14 quantitative figures under ignored `results/`.
  These candidates are labeled `sglsim-synthesis-derived`, not `final`.
- SGLSim runtime now emits AegisKV bookkeeping fields in `sim_stats` and
  `request_details.csv`: Host-visible tokens, GPUReplica candidate tokens, safe
  frontier tokens, recompute tokens, recovery action, and expected recovery
  time. The 20260621_030638 policy-candidate run validated these fields on
  96/96 completed request rows.
- SGLSim runtime now also emits online candidate failover fields in
  `request_details.csv`. Run `20260621_032719` completed 96/96 request rows,
  generated 24 critical-agent retry candidates, and measured 100.00%
  Host-backed candidate frontiers with 13.45 ms mean candidate failover
  overhead. These are what-if recovery events from observed online frontiers,
  not live request abort/retry executions.
- SGLSim failover replay smoke now exists as a benchmark-level next step. Run
  `20260621_042727` completed 120/120 rows, including 24 real HTTP retry replay
  rows marked `aegis_failover_replay=true`; all 24 replay rows were dispatched
  cross-rank with `failed_source_dp_rank != assigned_dp_rank`. All 24 were also
  bracketed by `/simulate/kill_dp` and `/simulate/restart_dp`: the recorded
  source scheduler subprocess was terminated before replay and restarted after
  replay, and every replay row is marked `aegis_hard_failure_injected=true`.
  All 24 replay rows also hit the runtime HostBackup directory
  (`aegis_host_backup_lookup_hit=true`), restoring 39,676 directory-backed
  tokens and recomputing 0 tokens. Of those protected tokens, 38,989
  prompt-prefix tokens were materialized as target-rank device-cache metadata
  before replay admission and appear as replay-row `cache_hit_tokens`.
  All 24 replay rows also found peer GPUReplica
  directory entries (`aegis_peer_gpu_lookup_hit=true`, 5,036 peer tokens); for
  this long-prefix workload the runtime lower-envelope still chose Host restore
  because peer GPUReplica would require recomputing most of the prefix. The
  scheduler now sleeps for the replay restore plan, with 13.51 ms mean observed
  restore sleep, so restore overhead affects online wall-clock latency. It
  validates the live serving path for killed-scheduler cross-rank retry-shaped
  requests with HostBackup and peer GPUReplica directory lookup plus SGLSim
  metadata page movement, but still does not execute real CUDA tensor KV copy.
- SGLSim physical-copy-calibrated restore smoke: run
  `sglsim-physical-copy-model-smoke-20260628` enables
  `SGLSIM_AEGIS_RECOVERY_MODEL=physical_copy` with the repeated physical copy
  aggregate from jobs `34338/34344/34345`. It completes 8/8 rows and 4/4
  hard-restart replay rows in the new `physical_copy_restore_smoke` scenario;
  all four replay rows are cross-rank, hit both HostBackup and peer GPUReplica
  directories, choose `gpu_replica`, and report 4.466 ms mean restore sleep.
  This validates the physical-copy-calibrated timing wiring, but it remains
  SGLSim metadata materialization rather than physical CUDA tensor KV movement.
- SGLSim recovery-action-calibrated repeated suite: run
  `sglsim-physical-recovery-action-agentic-r3-20260628` repeats the DP2 Host,
  DP4 peer, DP4 non-peer, and DP4 peer hot-tail hard-failover scenarios three
  times each under the recovery-shaped physical movement restore model. It
  completes 1,440/1,440 rows, 288/288 hard replay rows, and 0 failed rows. Mean
  restore sleep is 17.148 ms for DP2 Host, 16.914 ms for DP4 peer, 17.343 ms
  for DP4 non-peer, and 5.005 ms for DP4 peer hot-tail. Fig. 10 and Fig. 14 now use
  this suite for their online agentic rows, but the evidence tier remains
  preliminary because the runtime still materializes metadata instead of CUDA
  KV tensors.
- SGLSim live movement audit smoke: run `sglsim-live-movement-audit-smoke-20260628`
  verifies the new request-detail fields. It completes 8/8 rows and 4/4 replay
  restore rows; all restore rows report
  `aegis_replay_restore_data_plane=metadata_materialization` and
  `aegis_replay_restore_timing_source=physical_recovery_path_csv`. The strict
  audit intentionally fails until live recovery emits
  `physical_tensor_copy` rows.
- SGLSim repeated failover replay matrix scaffolding now exists as
  `scripts/run_failover_replay_matrix.py`. Run
  `sglsim-failover-replay-matrix-r5-20260621` completed 5 seeds with 600/600
  request rows and 0 failures. It generated 120 real HTTP retry replay rows,
  all cross-rank and all bracketed by hard scheduler kill/restart. All replay
  rows hit HostBackup and peer GPUReplica runtime directories, restoring
  201,787 HostBackup tokens, materializing 198,322 restored prompt-prefix
  tokens as target-rank cache metadata, and recomputing 0 tokens. Several
  preliminary SGLSim figure CSVs still use this r5 matrix, while Figs. 2, 3,
  8, 9, 10, 12, 13, 14, and 14b now use direct replay-calibrated or
  direct-sweep derivations layered on top of it and its short-prefix companion
  matrices. Fig. 11 appends the seven-job SGLang overhead aggregate.
- Maintainable agentic-workflow benchmark source has been restored under
  `benchmark/agentic_workflows/`. The wrapper names the smoke, DP2 shared
  prefix, DP2 hard-failover HostBackup, DP4 peer, and DP4 non-peer scenarios;
  generates per-trial synthesis configs; runs through the same SGLSim
  `benchmark/synthesis_requests` serving path; summarizes `request_details.csv`;
  and exports paper-facing candidate rows tagged
  `sglsim-agentic-online-preliminary`. This removes the missing-source
  reproducibility gap, but does not change the final-evidence gate.
- The restored wrapper was validated with an actual online smoke
  (`codex-smoke-20260621`, 4/4 rows completed) and an actual DP2 hard-failover
  HostBackup run (`codex-hard-host-20260621`, 120/120 rows completed, 24 hard
  replay rows, 24 HostBackup lookup hits, 38,868 restored tokens, 0 replay
  recompute tokens). The AegisKV experiment-facing entry
  `scripts/run_agentic_workflow_suite.py` can dry-run or execute suites and
  export combined online summaries plus paper-candidate CSVs.
- SGLSim repeated agentic suites `sglsim-agentic-online-r3-20260621`,
  `sglsim-agentic-peer-hot-tail-r3-20260621`, and
  `sglsim-physical-copy-agentic-r3-20260628` are retained as provenance. They
  have been superseded for Fig. 10 and Fig. 14 by
  `sglsim-physical-recovery-action-agentic-r3-20260628`, which keeps the same
  online hard-restart replay path while using recovery-shaped H2D/P2P movement
  curves for restore timing.
- The experiment-facing suite entry also completed a DP4 peer/non-peer route
  validation (`codex-peer-nonpeer-20260621`): both scenarios completed 120/120
  rows with 24 hard-failure replay rows; the peer-route scenario recorded
  24 peer GPUReplica lookup hits while the non-peer route recorded 0.
- Fig. 8 now uses an implemented online SGLSim policy sweep, not the older
  replay counterfactual. The active run id is
  `sglsim-policy-sweep-agentic-r6-dp4-matched-seed-20260628`: four explicit
  `gsp_aegis_policy` settings over the same DP4, 32-agent, 3-turn workload and
  matched seeds `8100/8101/8102`. Each policy completes 360/360 rows with 72
  hard-failure replay rows and 0 failures. Fig. 8 reports normalized WCT of
  1.000 no backup, 0.982 uniform Host, 0.982 uniform GPU, and 0.976 AegisKV.
- Fig. 9 and Fig. 10 still use the retained multi-policy replay-calibrated
  counterfactual `sglsim-fig8-10-policy-counterfactual-r11-20260621`:
  long-prefix Host r5 plus short-prefix peer r3 plus short-prefix non-peer
  recompute r3. Across 264 replay rows and 230,327 runtime recovery tokens,
  Fig. 9 reports mean/p95 recovery-latency
  ratios of 1.000/1.000 no backup, 1.110/1.257 uniform Host, 0.830/0.853
  uniform GPU, and 0.757/0.803 AegisKV; Fig. 10 reports equal-regime AegisKV
  path shares of 9.7% peer GPU, 33.3% Host H2D, 0.0% ordinary cache, and 56.9%
  recompute. Fig. 10 now also appends online agentic-suite rows: the base r3
  suite contributes three 100% Host-restore rows for DP2 HostBackup, DP4
  peer-route, and DP4 non-peer-route scenarios, while the peer hot-tail suite
  contributes a DP4 peer row with 19.2% peer-GPU restored tokens and 80.8%
  suffix recompute. Jobs `34338`, `34344`, and `34345` add a repeated raw
  physical H2D/D2H/P2P copy calibration anchor; jobs `34357`, `34358`, and
  `34359` add recovery-shaped Host/GPUReplica movement into retry KV buffers.
  Final evidence still needs live end-to-end physical CUDA KV-copy recovery and
  paired online policy sweeps.
- Fig. 2 now replaces the restore/recompute boundary with runtime
  replay-calibrated curves from the same 264 hard-failure replay rows. The
  active run id is `sglsim-fig2-restore-boundary-calibrated-r11-20260621`.
  The fitted SGLSim restore model is recompute `4.0 + 0.0108 * tokens` ms,
  Host `7.5 + 0.0036 * tokens` ms, and GPUReplica `2.4 + 0.0019 *
  peer_tokens + 0.0108 * suffix_tokens` ms, with 34.2% observed peer-token
  coverage in the GPUReplica regime. Jobs `34338`, `34344`, and `34345` add
  repeated real CUDA H2D/D2H/P2P microbench validation, but the final gate
  still needs those copy paths in the end-to-end recovery flow.
- Fig. 14 now preserves the plotted DP-scale dimension instead of accidentally
  overwriting it with provenance metadata from the source DP=2 replay matrix.
  It now goes one step further and uses a direct DP-size SGLSim serving sweep:
  DP=2 r5, DP=4 r3, and DP=8 r3. Those 11 runs completed 1,320/1,320 rows
  with 264 hard-failure replay rows and no failed requests. Because the
  workload is still low-concurrency and metadata-only, this is preliminary
  scale evidence, not final high-concurrency physical KV movement evidence.
- Fig. 14b now replaces the older replay derivation with an implemented online
  DP4 peer-route sweep. The active run id is
  `sglsim-peer-route-sweep-agentic-r1-dp4-matched-seed-20260628`: three
  retry-route policies run over the same short-tail workload and seeds
  `8500/8501/8502`. Non-peer routing produces 0/72 peer hits and 72 recompute
  actions; deterministic random healthy routing produces 21/72 peer hits; and
  AegisKV peer routing produces 72/72 peer hits and 72 GPUReplica actions. The
  current AegisKV peer route is 0.988x normalized WCT and 74.3% p99 recovery
  overhead relative to the non-peer recompute route. This is direct online
  SGLSim evidence for retry-rank locality, but still not a final physical P2P
  KV-copy policy sweep.
- Fig. 15 now uses a real mixed calibration anchor instead of synthetic
  SGLang coordinates: SGLang HiCache write-through p99 overhead is 0.31% across
  the seven matched DP=2 Slurm runs, while the online SGLSim AegisKV
  no-failure sweep reports -0.49% p99 overhead with 0.93% 95% CI. This puts the
  normal-path overhead anchor in the noise range on both systems; final paired
  policy ranking evidence is still missing.

机制图已经可用于设计审阅和早期论文版式。定量图也已经比纯模板更有用，因为它们来自 SGLSim serving-derived candidate CSV、online agentic workflow trace、SGLang paired anchor、torch/CUDA physical-copy calibration 或 base SGLang/SGLSim accuracy run，并由相应 manifest 和 audit 脚本支撑。当前只有 Fig. 10b 具备 strict-final-ready 定量证据；其余 21 张定量图仍保持 preliminary / hybrid-ready，直到被带 physical KV movement 的 final online AegisKV policy sweep、paired SGLang measurement 或明确的 final motivation evidence 替换。

当前 validator 结果：

```text
figures=28
claim_ready=7
result_figures=24
result_figures_with_preliminary_data=21
result_figures_without_final_measured_data=21
```

## 仍缺少的论文强度证据

1. SGLang small-scale evidence on `hkbugpumanode`.
   - DP=2 real SGLang run through Slurm is smoke-validated by job `32855`.
   - No-failure baseline vs HiCache write-through overhead now has seven matched
     paired data points from jobs `32861`, `32863`, `32869`, `32870`, `32871`,
     `32874`, and `32899`; their aggregate is included directly in Fig. 11 as
     `sglang-small-paired` evidence, but still needs representative AegisKV
     policy workloads before it can be final Fig. 11 evidence.
   - Failure retry continuity is now smoke-validated by jobs `32977` and
     `32978`: 4/4 in-flight routed requests survived DP-rank failure injection
     and returned `sglext.failover` responses in both runs. Job `32978` proves
     the same with 760 pre-failover output tokens, but no storage backend.
   - Host-visible frontier reporting is now smoke-validated by job `32980`:
     enabling SGLang `remote_backup` produces 5,132 pre-failover backed-up
     tokens across 4 failover-retried requests. The derived
     `failure_visibility_pair_32978_32980` artifact pairs this against job
     `32978`, which has 0 backed-up tokens, and records 1.108x p99 latency for
     the storage-backed smoke.
   - Remote-backup storage matching and single-request materialization are now
     smoke-validated by jobs `32982` and `32983`: the former sees 1,297
     retry-side storage pages and `prefetch_get` hits, while the latter uses
     `wait_complete` to turn those pages into 1,297 retry `#cached-token`
     tokens. Job `32984` supplies the no-storage single-request control, job
     `32985` shows the four-request split-prefix case still cannot restore
     backed pages, and jobs `32987`/`32989` show that four-request concurrent
     restore does work when each prompt has page-0 backed data. Job `32991`
     resolves the four-request shared-prefix materialization gap after the
     split-prefix storage fix. Jobs `32993`/`32994` extend the same
     visibility check to a longer shared-prefix frontier, but the visible
     remote-backup path is still slower. The new physical Host-restore parser
     makes these states auditable as CSV rows rather than prose-only evidence.
     The remaining final gap is running
     paired Host-visible/Host-invisible restore-benefit and physical copy
     comparisons under matched workload cost and representative shared-prefix
     concurrency, at larger scale.
   - Metadata overhead, fixed write-through D2H ACK latency, H2D restore latency, GPUReplica/P2P copy latency.
   - Failure retry path with Host-visible vs Host-invisible comparison.
   - No-failure p95/p99 overhead under identical fixed write-through D2H for
     full workloads and repeated trials.

2. SGLSim AegisKV online implementation evidence.
   - Visible `benchmark/agentic_workflows` source has been rebuilt as a thin
     wrapper over `benchmark/synthesis_requests`, with dry-run config
     materialization, suite execution, run summarization, and paper-candidate
     CSV export.
   - `benchmark/synthesis_requests/run.py` is smoke-validated as the current
     maintainable online benchmark entry, but it does not yet implement the
     AegisKV policies.
   - A SGLSim serving-derived policy candidate path now exists and validates
     figure schemas from real `request_details.csv`, but the AegisKV policy is
     only partly online: request metadata, frontier/protection metrics,
     candidate failover estimates, and benchmark-level killed-scheduler
     cross-rank retry replay are emitted by the runtime/client, while true
     physical Host-visible or peer KV page copy is still not executed as an
     online recovery path.
   - Physical KV movement must run in the online simulator path before these
     results can become final evidence.
   - A repeatable hard-failover replay matrix runner now refreshes the current
     `sglsim-preliminary` CSVs. Fig. 14 additionally has a direct DP=2/4/8
     SGLSim serving sweep, Fig. 11 now has a paired online no-failure overhead
     sweep, and Fig. 10b now has a final broad shared-prefix sweep. Large final
     policy sweeps must still replace Fig. 10, Fig. 11, and Fig. 14 with
     `evidence_tier=final`.

3. Calibration evidence.
   - Fig. 15 now has a real normal-path overhead anchor, but paired
     SGLang/SGLSim policy runs are still required for final calibration.
   - The acceptable final claim is ranking preservation, not exact latency
     equality.

4. Correctness evidence.
   - Unit-level SGLSim/SRT metadata tests now cover pending or missing
     HostBackup rejection, unsupported pending restore action rejection,
     `lock_ref > 0` eviction protection, divergent radix insert after restore,
     Host/peer frontier invalidation, peer-rank demotion, and baseline requests
     without Aegis metadata.
   - Evidence note: `experiments/correctness_evidence.md`.
   - Remaining final gap: the same correctness envelope still needs real SGLang
     failure retry and physical KV-copy validation.

## 当前 HPCA readiness 判断

论文叙事、问题建模、系统设计、图表计划和视觉资产已经足够支撑内部 paper review draft。

在 calibrated-evidence 标准下，当前草稿已经达到 HPCA-hybrid-ready：`validate_figure_evidence.py --strict-hpca-hybrid` 会将 25/25 张图识别为 ready。其中 Fig. 1/1b/1c/3d 属于 trace-derived motivation，Fig. 3b 属于 `hybrid-model-policy`，Fig. 3c 属于 `hybrid-online-alpha-sweep`，Fig. 3/8/12/13/14b 属于 `hybrid-online-policy-sweep`，Fig. 9/11/11b/15 属于 `hybrid-paired-anchor`，Fig. 2/10/14/15b/15c/15e 属于 `hybrid-physical-calibrated`，Fig. 15d 属于 `hybrid-simulator-accuracy`，Fig. 15f 属于 `hybrid-sglang-host-restore`。physical-calibrated 图由 recovery-shaped physical CUDA microbench jobs `34357/34358/34359` 和 P0 physical bridge coverage 审计锚定，Fig. 15f 由真实 SGLang Host restore parser rows 锚定；但 live SGLSim rows 仍是带 calibrated timing 的 `metadata_materialization`，不是 live CUDA tensor movement。

剩余投稿缺口是 strict-final evidence：大多数定量图仍使用 preliminary SGLSim metadata-materialization row，因此最终论文 claim 仍需要带 physical KV movement 的端到端测量，或者显式验证过的真实系统恢复路径。

投稿 gate：

```bash
python benchmark/AegisKV/draft/scripts/validate_draft_integrity.py
python benchmark/AegisKV/draft/figures/scripts/validate_figure_evidence.py --write-hpca-hybrid
python benchmark/AegisKV/draft/figures/scripts/validate_figure_evidence.py --strict-final
python benchmark/AegisKV/draft/figures/scripts/validate_figure_evidence.py --strict-hpca-hybrid
```

只有这些命令全部通过后，草稿才可以声称是完整 HPCA submission candidate。当前状态下，draft integrity 和 hybrid-status generation 通过；两个 strict gate 仍会按预期失败。
