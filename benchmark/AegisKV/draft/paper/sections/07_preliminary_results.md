# 07. 初步结果与图注

This section is the current result narrative for the draft. It is intentionally
split into two layers:

1. **Paper narrative.** What each figure is meant to prove and how the results
   should be read.
2. **Evidence maturity.** Whether the current artifact is final evidence or a
   preliminary SGLSim counterfactual used for internal paper development.

At the time of this draft, most quantitative result figures are backed by
preliminary measured CSVs refreshed from SGLSim hard-failover replay matrices,
direct replay-derived counterfactuals, and small SGLang overhead smokes. The
only current quantitative strict-final-ready figure is Fig. 10b, which uses a
dedicated 3-run online SGLSim shared-prefix sweep with low/mid/high overlap
scenarios. Fig. 1 is now treated as hybrid-ready trace-derived motivation,
not strict-final real-system evidence. The older policy-model SGLSim rows are
labeled:

```text
evidence_tier=sglsim-synthesis-derived
source=sglsim_synthesis_request_details_policy_model_repeated
```

Newer rows use narrower labels such as `sglsim-runtime-replay-derived` or
`sglsim-runtime-replay-calibrated` when they are computed directly from replay
request details. These plots are stronger than hand-drawn expected trends
because they are reproducible from real SGLSim serving `request_details.csv`
rows, aggregate repeated runs with std/95% CI where available, and are
validated by `figures/scripts/validate_figure_evidence.py`. Fig. 11 now uses
`sglsim-agentic-online-preliminary` rows from
`codex-normal-overhead-online-r3-20260628` for the baseline/AegisKV online
no-failure comparison, while retaining replay-derived uniform Host/GPU
comparators. It also contains `sglang-small-paired` rows from seven matched
DP=2 SGLang overhead jobs on `hkbugpumanode`, including larger 64-request and
128-request / 128-token repeats. Jobs `32977` and `32978` additionally provide
DP=2
real-SGLang failure-control-path smokes: in both runs, 4/4 routed in-flight
requests were captured, re-dispatched, completed, and returned
`sglext.failover` after `/simulate_gpu_failure`. Job `32980` adds the first
storage-backed SGLang smoke by enabling `remote_backup`; it reports 5,132
pre-failover backed-up tokens across 4 failover-retried requests. The paired
artifact `failure_visibility_pair_32978_32980` now compares job `32978`
against job `32980`: the Host frontier increases from 0 to 5,132 backed-up
tokens, while p99 latency normalizes to 1.108x. This supports frontier
visibility, not final restore-latency benefit. Two follow-up probes refine the
remaining gap. Job `32981` disables request-aware remote-backup retention and
still sees retry prefetch revoked with `data_pages=0`, ruling out lease GC as
the main miss cause. Job `32982` uses a single request, writes the long prompt
from `page_start=0`, and the retry rank sees `data_pages=1297`,
`Prefetch ACCEPTED`, and `prefetch_get` hits, but the default `best_effort`
policy terminates before IO completion and admits with `#cached-token: 0`.
Job `32983` reruns the same single-request shape with
`hicache_storage_prefetch_policy=wait_complete`; it completes prefetch IO with
`completed=1297 / expected=1297` and retry prefill admits with
`#cached-token: 1297`. Job `32984` adds the matching no-storage single-request
control; the resulting `failure_visibility_pair_32984_32983` artifact records
a 1,297-token Host frontier delta and 1.124x visible/no-storage p99 latency
ratio. Job `32985` then reruns the four-request remote-backup case with
`wait_complete`; it still reports `data_pages=0` for all retry prefetches,
confirming that the concurrent shape needs split-prefix-aware restore rather
than only a different prefetch stop policy. Job `32987` removes that shared
prefix by enabling `UNIQUE_REQUEST_PREFIX=1` and restores four concurrent
requests successfully: all four retry queries reach `data_pages=1301`,
all four prefetch IOs complete `1301 / 1301` tokens, and each retry prefill
admits with `#cached-token: 1301`. Job `32989` is the matching no-storage
unique-prefix control, and the `failure_visibility_pair_32989_32987` artifact
records a 5,204-token Host frontier delta with visible/no-storage p99 latency
ratio 1.308x. Job `32991` applies the split-prefix storage fix and reruns the
shared-prefix four-request shape with `remote_backup + wait_complete`; all four
retry queries reach `data_pages=1297`, all four prefetch IOs complete
`1297 / 1297`, and each retry prefill admits with `#cached-token: 1297`. The
new `failure_visibility_pair_32992_32991` artifact compares the same-code
no-storage shared-prefix control against the patched storage-backed run: the
Host frontier increases from 0 to 5,188 backed-up tokens with a 1.210x
visible/no-storage p99 latency ratio. Jobs `32993`/`32994` repeat the paired
shape with a longer shared prefix (`PROMPT_REPEAT=256`) and extend the visible
frontier to 20,548 backed-up tokens. The visible run materializes all four
retries with `data_pages=5137` and retry `#cached-token: 5137`, but the visible
p99 latency is still 1.630x the no-storage control. Jobs `32998`/`32999` then
isolate failure-time visibility under matched normal write cost: both jobs
write the same 20,548 remote-backup tokens, while `32998` disables retry
storage prefetch and `32999` materializes 5,137 cached tokens per retried
request. The visible run remains slower at 1.487x the lookup-disabled control,
so the new pair strengthens the Fig. 11b visibility mechanism evidence without
making it final recovery-benefit evidence. Jobs `33002`/`33003` rerun the same
matched-write-cost shape with `best_effort` instead of `wait_complete`: both
write remote-backup tokens, the visible retry path finds `data_pages=5137`, and
prefetch partially materializes before retry prefill. The repeated pairs
`33002/33003`, `34333/34334`, and `34336/34337` now aggregate to 0.961x
visible/control p99 with 0.016 95% CI, and all three repeats improve p99. This is the first repeated
SGLang same-write-cost latency-improvement point, but it is partial
asynchronous restore, not final full restore policy evidence. Final restore
benefit now depends on paired
Host-visible/Host-invisible AegisKV policy measurements under matched
normal-path cost at larger scale, not on the earlier split-prefix correctness
gap.
The physical Host-restore evidence is now also machine-readable:
`sglang_physical_restore_evidence_summary.md` and
`results/physical_restore_evidence/physical_restore_runs.csv` parse the real
SGLang logs for `data_pages`, accepted prefetch pages, completed prefetch
tokens, and retry `#cached-token`. This turns the Host-restore chain into
structured evidence: `32998/33002` are hidden-frontier controls,
`32983/32987/32991/32999` are full materialization points, and
`33003/34334/34337` are partial best-effort asynchronous materialization
points. Jobs `34338/34344/34345` add a repeated physical CUDA copy calibration:
on two NV12-connected A100s, an 8192-token KV segment takes 48.595 ms H2D
(95% CI 0.531 ms), 46.416 ms D2H (95% CI 0.100 ms), and 4.368 ms
GPU0-to-GPU1 P2P (95% CI 0.004 ms) at p50 under the Qwen3-8B bf16 KV-size
assumption.
Jobs `34357/34358/34359` then reshape this into AegisKV backup/restore actions
that write into retry-rank KV buffers. At 8192 tokens, `host_restore_h2d` takes
48.434 ms p50 (95% CI 0.013 ms), while `gpu_replica_restore_p2p` takes 4.330 ms
p50 (95% CI 0.005 ms). This is stronger than raw copy calibration, but remains
an action-level microbenchmark rather than live end-to-end policy recovery.
Except for Fig. 10b, these data are not final HPCA evidence. Final
quantitative claims require replacing the remaining preliminary CSVs with
`evidence_tier=final` data from online AegisKV SGLSim runs with physical KV
movement and paired small-scale SGLang measurements.

## 证据概览

| Artifact class | Count | Current status | Final gate |
|---|---:|---|---|
| generated system overview | 1 | `design-ready` bitmap + prompt | visual/content review |
| quantitative figures | 24 | 1 strict-final measured, 23 preliminary / hybrid-ready | replace remaining preliminary rows with `evidence_tier=final` or explicitly keep them as motivation / calibration evidence |
| tables | 4 | design/evaluation scaffolding | align with final implementation |
| SGLang smoke | multiple | DP=2 Qwen3-8B Slurm smoke, paired baseline vs HiCache write-through overhead smoke, DP=2 failover-control smoke, remote-backup frontier smoke, materialized Host restore smoke, same-write-cost best-effort repeats, and CUDA H2D/D2H/P2P microbench | full AegisKV paired runs at larger scale with physical copy in the recovery path |
| SGLSim runs | multiple | local `synthesis_requests` smoke, 96-row policy-candidate run, hard-failover replay smoke, 5-seed hard-failover replay matrix, DP=2/4/8 direct scale sweep, DP=4 matched peer-route replay pair, Fig. 8 online policy sweep, Fig. 11 online no-failure overhead sweep, trace-derived Fig. 1 equal-size loss motivation, and strict-final-ready Fig. 10b shared-prefix sweep | online AegisKV policy matrix for remaining figures |

Current validator summary:

```text
figures=25
claim_ready=2
result_figures=24
result_figures_with_preliminary_data=23
result_figures_without_final_measured_data=23
hpca_hybrid_ready=25/25
hpca_hybrid_ready_result_figures=24/24
hpca_hybrid_blocked_result_figures=0/24
```

The hybrid evidence report is generated with:

```bash
python benchmark/AegisKV/draft/figures/scripts/validate_figure_evidence.py --write-hpca-hybrid
```

It marks Fig. 2, Fig. 10, and Fig. 14 as
`hybrid-physical-calibrated`: the large-scale rows are still SGLSim
metadata-level simulation, but their recovery timing is anchored by the
recovery-shaped physical CUDA movement jobs `34357/34358/34359`.

The hard submission gate remains:

```bash
python benchmark/AegisKV/draft/figures/scripts/validate_figure_evidence.py --strict-final
python benchmark/AegisKV/draft/figures/scripts/validate_figure_evidence.py --strict-hpca-hybrid
```

The per-figure path from preliminary artifact to final evidence is tracked in
`experiments/final_evidence_matrix.csv`.

## 结果叙事

### 动机：价值不等于字节大小

Fig. 1 holds lost KV size fixed and varies the lost segment's workflow role.
The final CSV injects the same 512-token loss into 36 observed online SGLSim
workflow timelines from `codex-shared-prefix-sweep-r3-20260628`. Losing a
terminated branch is nearly free (0.001 normalized WCT delta), running subagent
loss is mostly hidden by parallel slack (0.005), and critical positions become
progressively more expensive: last subagent before fan-in 0.020, main resume
0.037, synthesis tail 0.051, and shared prefix 0.147. The claim is that the
right optimization target is **workflow completion time impact per recoverable
token**, not copied bytes or raw reuse count.

### 恢复动作存在下包络

Fig. 2 compares recompute, Host restore, and GPUReplica restore as segment
length changes. The current curves are calibrated from the same hard-failure
replay rows used by Figs. 9-10: recompute rows fit `4.0 + 0.0108 * tokens` ms,
Host rows fit `7.5 + 0.0036 * tokens` ms, and GPUReplica rows fit a peer-tail
restore plus suffix recompute model with 34.2% observed peer coverage. Recompute
is competitive for tiny suffixes or non-peer retries because it avoids restore
setup overhead. GPUReplica is fastest for small hot tails when peer pressure is
low. Host restore becomes attractive for long prefixes where H2D is cheaper
than replay. AegisKV should therefore choose a recovery action per segment
rather than treating every backup hit as "restore all." The first physical
copy calibration aggregate, SGLang small-scale jobs `34338/34344/34345`,
measures real CUDA KV-sized movement on two NV12-connected A100s: for an
8192-token segment under the 147,456 bytes/token Qwen3-8B bf16 assumption,
p50 latency is 48.595 ms for H2D, 46.416 ms for D2H, and 4.368 ms for
GPU0-to-GPU1 P2P.

### 静态 GPUReplica budget 很脆弱

Fig. 3 sweeps the GPUReplica HBM budget fraction. Different workload regimes
expose different failure-time visibility limits: hot-tail rows improve from
the budget=0 baseline and then saturate once the peer-visible tail is covered;
long-prefix rows are already better served by Host restore, so forcing a peer
split can add setup cost; fan-out/non-peer rows do not improve with HBM budget
alone because the retry rank has no peer-visible tokens. Current data is an
implemented DP4 online hard-restart sweep:
`sglsim-budget-sweep-agentic-r1-combined-20260628`, with 25 budget/timing
scenarios, 600 replay rows, and 0 failed rows.

Fig. 3b and Fig. 3c now make the adaptive-alpha path explicit. Fig. 3b is
generated from the offline value-iteration policy table
`aegiskv_alpha_mdp_policy.json`, whose state space is
`critical_pressure in {low, medium, high}` crossed with five
`failure_rate_per_window` values. The policy uses `gamma=0.92` and 50 value
iterations, and the selected `alpha` is nondecreasing as failure risk rises
within each pressure bin. The goal audit replays all 15 policy-table states
through the runtime function `sglsim.srt.simulation.alpha_policy.choose_alpha`
and verifies exact agreement with the JSON table and Fig. 3b CSV.

Fig. 3c then verifies that the same policy table is used during online serving.
The 1-3-1-3-1 sweep contains four online runs, low/high workload pressure, and
failure rates 0.003, 0.01, and 0.03. The raw traces contain 150 Critical
requests and 120 Non-Critical requests. Replaying `choose_alpha` over all
270 completed request rows exactly matches the recorded
`aegis_gpu_budget_fraction`. Critical requests receive the selected
GPUReplica alpha, while Non-Critical requests retain Host visibility but have
zero effective GPUReplica tokens. This is stronger than a plot-only claim: it
checks the offline policy JSON, the runtime selector, the measured CSV, and
the online `request_details.csv` rows as one chain.

### 系统路径：在 SGLang KV 语义上构建 policy

Figs. 4-7b define the system path. AegisKV is not a replacement KV manager. It
adds workflow metadata, HostBackup visibility, peer placement, and safe-frontier
selection on top of SGLang radix/HiCache lifecycle events. The correctness
figures emphasize that pending D2H is not safe, `lock_ref > 0` prevents
eviction, ACK creates the recoverable frontier, and Host/peer eviction must
invalidate recovery choices. Current unit evidence is recorded in
`experiments/correctness_evidence.md` and covers the SGLSim/SRT metadata
invariants behind these states; real SGLang failure/copy validation remains
future final evidence.

### 整体 WCT 与恢复行为

Fig. 8 now uses an implemented online SGLSim policy sweep rather than the older
replay counterfactual. The active run is
`sglsim-policy-sweep-agentic-r6-dp4-matched-seed-20260628`: DP4, 32 agents,
3 turns, matched seeds `8100/8101/8102`, and four explicit
`gsp_aegis_policy` settings. Each policy completes 360/360 rows with 72
hard-failure replay rows and 0 failures. Normalized mean end-to-end WCT is
1.000 for no backup, 0.982 for uniform Host, 0.982 for uniform GPUReplica, and
0.976 for AegisKV. Fig. 9 isolates recovery overhead using the same online
policy sweep: AegisKV is 0.780x no-backup recompute on mean retry latency and
0.915x on critical-agent p90 latency, compared with 0.793x/0.908x for uniform
Host and 0.829x/0.935x for uniform GPUReplica. Fig. 10 explains the replay path
accounting across the retained multi-regime counterfactuals: AegisKV combines
9.7% peer GPU, 33.3% Host H2D, and
56.9% suffix recompute under the current equal-regime mix. Fig. 10 also
includes repeated online action rows from
`sglsim-physical-recovery-action-agentic-r3-20260628`, which uses the
recovery-shaped physical movement aggregate for restore timing. Across 12
trials and 288 hard-failover replay rows, all requests complete with no
failures. The DP2 Host and DP4 peer long-prefix scenarios split between Host restore and
small peer-GPU hot-tail restore; the DP4 non-peer scenario is 100% Host
restore; the DP4 peer hot-tail scenario chooses `gpu_replica` for 72/72 replay
rows and splits into 19.2% peer-GPU restored tokens plus 80.8% suffix
recompute. This is still not final physical-movement evidence:
`sglsim-live-movement-audit-smoke-20260628` records 4/4 restore rows with
`aegis_replay_restore_data_plane=metadata_materialization` and
`aegis_replay_restore_timing_source=physical_recovery_path_csv`. Strict live
movement validation now requires `physical_tensor_copy` rows, but that should
be collected through a SGLang/torch real-system path rather than by adding CUDA
movement to the metadata-only SGLSim runtime.

### 避免重复保护

Fig. 10b evaluates shared-prefix marginal accounting. When many agents share a
long prompt prefix, naive Host/GPUReplica accounting can count the same
safe-frontier content multiple times. AegisKV's marginal accounting protects
only newly extended frontier tokens within the same agent. The final Fig. 10b
CSV comes from `codex-shared-prefix-sweep-r3-20260628`: nine online SGLSim
runs across low/mid/high shared-prefix scenarios, 141 multi-turn agent groups,
and shared-prefix bins from 2.5% to 77.5%. The derivation excludes retry rows,
so retry cache-hit counters do not define the shared-prefix ratio.

### Normal-path overhead 与 Host visibility

Fig. 11 separates no-failure overhead from failure recovery benefit. The paper
must compare AegisKV against a fixed write-through baseline that pays the same
D2H path. Fig. 11b captures the key distinction: Host-invisible baseline pays
D2H but cannot use failed-rank Host KV after a GPU failure, while AegisKV adds
cross-rank HostBackup directory visibility and can restore ACKed frontiers.
The current Fig. 11b retry-latency axis is now tied to direct SGLSim replay
counterfactuals over recoverable regimes: Host-visible long prefixes and
peer-visible hot tails. Its overhead axis remains a preliminary fixed
write-through / peer-copy anchor.
The matched SGLang DP=2 overhead smokes on `hkbugpumanode` support this
direction. Job `32861` measured 8/8 successful requests in both baseline and
HiCache write-through cases, with 2.02% p99 overhead and 3.05% mean overhead.
Job `32863` increased the workload to 32 requests and 128 max tokens per case;
it measured 1.18% mean overhead, while p99 was -3.93% due small-sample tail
noise. Together these points support a bounded-overhead direction, not a final
precise p99 claim.
Jobs `32977` and `32978` add real SGLang failure-control-path smokes. In both
runs, 4 routed in-flight requests on DP rank 0 were captured by
`/simulate_gpu_failure`, re-dispatched to the surviving rank, completed
successfully, and returned `sglext.failover`. Job `32978` triggers failure
later and captures 760 pre-failover output tokens, but it still records 0
pre-failover backed-up tokens because no storage backend is attached. Job
`32980` attaches the `remote_backup` storage backend and reports 5,132
pre-failover backed-up tokens across the same four-request shape. That is a
real Host-visible frontier smoke. The derived
`failure_visibility_pair_32978_32980` artifact makes the comparison explicit:
job `32978` has 0 backed-up tokens, job `32980` has 1,283 backed-up
tokens/request, and the remote-backup smoke is 1.108x the no-storage p99
latency. Job `32981` shows that disabling request-aware remote-backup retention
does not fix the miss: four-request retries still query from page 0 while the
backed leaf data begins at page 14. Job `32982` removes that split-prefix
shape with a single request and validates the next step in the chain:
`data_pages=1297`, `Prefetch ACCEPTED`, and remote `prefetch_get` hits after
failover. Job `32983` then switches that single-request case to
`wait_complete`, producing `Prefetch IO END ... completed=1297 / expected=1297`
and retry prefill `#cached-token: 1297`. Job `32984` supplies a no-storage
single-request control for that case; the pair records 1.124x visible/no-storage
p99 latency, so it is correctness/calibration evidence rather than a latency
win. Job `32985` shows that four concurrent retried requests still have
navigable remote radix state but `data_pages=0`, so final Fig. 11b still needs
split-prefix-aware restore for shared-prefix prompts. Job `32987` proves that
four concurrent requests do materialize when the prompt shape starts from
page 0 per request, and job `32989` supplies the no-storage control. Job
`32991` then fixes the shared-prefix storage layout and materializes all four
shared-prefix retries with `data_pages=1297`, `completed=1297 / expected=1297`,
and retry `#cached-token: 1297`. The `failure_visibility_pair_32992_32991`
artifact records a 5,188-token Host frontier delta and a 1.210x
visible/no-storage p99 latency ratio. Jobs `32993`/`32994` extend the same
Host-frontier calibration to a longer shared prefix: the visible run backs up
20,548 tokens and materializes `#cached-token: 5137` on all four retries, but
its p99 remains 1.630x the no-storage control. Jobs `32998`/`32999` add a
fixed-write-cost pair: both write 20,548 tokens, but only the visible run is
allowed to use the storage frontier during retry, materializing
`#cached-token: 5137`; p99 remains 1.487x the lookup-disabled control. Jobs
`33002`/`33003`, `34333`/`34334`, and `34336`/`34337` show the corresponding
`best_effort` tradeoff: partial restore improves p99 in all three repeats and
aggregates to 0.961x visible/control p99 with 0.016 95% CI, but does not guarantee complete
materialization. The physical-restore parser records these same states as
structured rows, including full materialization in `32999`, partial
materialization in `33003/34334/34337`, and hidden-frontier controls in
`32998/33002/34333/34336`. Final Fig. 11b still needs a
paired AegisKV policy comparison that demonstrates restore-latency benefit
under matched normal-path cost and larger workloads.

### Ablation、sensitivity、scale 与 calibration

Fig. 12 removes AegisKV signals one at a time using implemented online toggles.
The DP4 hot-tail sweep runs matched seeds for full AegisKV, no lifecycle, no
segment granularity, no restore/recompute choice, no pressure signal, no
hysteresis, and no critical-agent restore. AegisKV is normalized to 1.0, all
disabled-signal variants regress, and the request rows expose the actual
restore actions selected by the retry path. Fig. 13 now uses an implemented
online failure-timing sweep over early/mid/late decode failures with the same
GPUReplica budget points as Fig. 3. Fig. 14
now uses a direct SGLSim
hard-failover replay DP-size smoke at DP=2, DP=4, and DP=8. Under the current
low-concurrency workload,
duration is nearly flat across DP sizes; this validates the serving path across
larger DP topologies but does not yet prove high-concurrency scale benefit.
Fig. 14b now uses an implemented DP=4 online peer-route sweep with explicit
retry route policies over the same short-tail workload and matched seeds
`8500/8501/8502`. The non-peer route has 0/72 peer hits and 72 recompute
actions; deterministic random healthy routing has 21/72 peer hits, 21
GPUReplica actions, and 51 recompute actions; AegisKV peer routing has 72/72
peer hits and 72 GPUReplica actions. AegisKV peer routing normalizes to 0.988x
WCT and cuts p99 recovery overhead to 74.3% of the non-peer recompute route.
This supports retry-locality in peer selection but is still not a final
physical P2P KV-copy policy sweep. Fig. 15 now
uses the defensible calibration anchor available today: real SGLang
normal-path overhead from the seven matched DP=2 paired runs versus the online
SGLSim no-failure AegisKV overhead sweep. Both anchors are in the noise range;
the final acceptable claim remains ranking preservation between SGLang and
SGLSim, not exact latency equality.

The current SGLSim preliminary path has progressed beyond a 96-row
policy-candidate smoke. The current r5 hard-failover replay matrix completed
5 seeds, 600/600 request rows, and 120 hard-failure cross-rank HTTP retry
replay rows.
All replay rows hit HostBackup and peer GPUReplica runtime directories,
restored 201,787 HostBackup tokens, materialized 198,322 prompt-prefix tokens
as target-rank metadata cache, recomputed 0 tokens, and observed 13.62 ms mean
restore sleep. The active figure CSVs aggregate the five run-level candidate
derivations as mean/std/95% CI. This is useful for debugging figure schemas,
policy equations, and paper layout; it still does not satisfy the final online
policy evidence gate.

The restored agentic workflow wrapper now also has a repeated online suite:
`sglsim-agentic-online-r3-20260621` runs DP2 shared-prefix normal path, DP2
HostBackup hard-failover, DP4 peer, and DP4 non-peer scenarios for three trials
each. The suite completes all rows with no failures: 288/288 normal-path rows
and 360/360 rows per hard-failover scenario. The newer
`sglsim-physical-recovery-action-agentic-r3-20260628` suite reruns the Host,
peer, non-peer, and hot-tail peer scenarios with the repeated recovery-shaped
physical movement aggregate driving restore timing. It completes 1,440/1,440
rows and 288/288 hard replay rows. Mean restore sleep is 17.148 ms for DP2
Host, 16.914 ms for DP4 peer, 17.343 ms for DP4 non-peer, and 5.005 ms for
DP4 peer hot-tail. Fig. 10 now uses these recovery-action-calibrated online rows for action coverage, and Fig.
14 consumes the Host/peer/non-peer rows as two preliminary online workflow scale
series. They remain `sglsim-agentic-online-preliminary` until final policy
sweeps replace the current metadata-only rows. The audit script
`audit_live_physical_movement.py` now makes this boundary machine-checkable.

Fig. 14 is the current exception to the r5-derived policy-candidate path. It
uses a direct DP-size sweep: DP=2 reuses the 5-seed r5 matrix, while DP=4 and
DP=8 add 3 seeds each. Across those 11 serving runs, 1,320/1,320 rows complete,
264 hard-failure replay rows execute cross-rank, and normalized duration is
1.000 at DP=2, 0.985 at DP=4, and 0.983 at DP=8. This is a direct scale smoke,
not final high-concurrency scale evidence.

Fig. 14b is also direct online SGLSim evidence rather than a policy-model row.
`sglsim-peer-route-sweep-agentic-r1-dp4-matched-seed-20260628` executes three
retry-route policies: forced non-peer, deterministic random healthy rank, and
recorded-peer routing. The figure plots the observed route hit/miss behavior
from those online replay rows, including the random route's measured 29.2%
peer-hit rate.

Fig. 15 is no longer a synthetic SGLang-vs-SGLSim ranking scatter. It now
compares current normal-path p99 overhead anchors: baseline is 0.00% in both
systems, while the protected path is 0.31% for SGLang HiCache write-through and
-0.49% for the online SGLSim AegisKV no-failure sweep, with 0.93% 95% CI on
the SGLSim point. The useful claim is that current normal-path overhead is
within noise in both systems; final ranking calibration still needs matched
SGLang/SGLSim policy workloads.

## HPCA 风格图注

**Fig. 1. Workflow penalty of equal-size KV loss.** Losing the same number of
KV tokens has very different impact depending on where the segment sits in the
agent workflow. The plot motivates segment-level value modeling instead of
byte-level backup rules. Current data: final SGLSim online equal-size loss
injection over 36 workflow groups, using a fixed 512-token loss.

**Fig. 2. Restore-vs-recompute boundary.** Recompute, Host restore, and
GPUReplica restore dominate in different segment-size regimes. AegisKV follows
the lower envelope rather than always restoring a backup hit. Current data:
SGLSim runtime replay-calibrated curves from 264 hard-failure replay rows,
plus repeated SGLang physical H2D/P2P CUDA-copy calibration curves from jobs
`34338/34344/34345`.

**Fig. 3. Static GPUReplica budget gap.** A single fixed peer-GPU backup budget
does not match all workloads. Static peer HBM helps only when protected tokens
are visible at the retry rank; otherwise Host restore or recompute still wins.
Current data: SGLSim runtime replay-calibrated counterfactual from 264
hard-failure replay rows.

**Fig. 4. AegisKV system overview.** Workflow metadata is attached to SGLang
requests, propagated to radix segments, and used to choose HostBackup
visibility, GPUReplica placement, and failure-time restore/recompute actions.
Current artifact: design-ready.

**Fig. 5. Backup data path.** HostBackup and GPUReplica have different
latency, capacity, queueing, and failure-domain semantics. Current artifact:
design-ready.

**Fig. 5b. Fixed HostBackup visibility pipeline.** Baseline and AegisKV both
pay fixed write-through D2H. AegisKV's Host benefit comes from publishing
ACKed failed-rank Host frontiers to retry ranks. Current artifact:
design-ready.

**Fig. 6. Online decision loop.** The controller observes workflow/KV/resource
state, chooses placement, waits for physical ACKs, publishes safe frontiers,
and uses them during failure recovery. Current artifact: design-ready.

**Fig. 7. Failure recovery timeline.** AegisKV shortens retry by using ACKed
frontiers and recomputing only the unsafe suffix. Current artifact:
design-ready.

**Fig. 7b. Radix lock, ACK, and eviction states.** Safe frontiers respect
SGLang radix correctness: locked nodes are not evicted, pending copies are not
safe, and Host/peer eviction invalidates recovery coverage. Current artifact:
design-ready, with matching unit-level metadata checks in
`test/unit/srt/test_aegis_restore_lookup.py`.

**Fig. 8. Overall workflow completion time.** Current data comes from direct
multi-regime replay counterfactuals: 264 hard-failure replay rows and 230,327
recovery tokens. AegisKV is 0.972x the same-row no-backup recompute
counterfactual on this low-concurrency smoke workload, compared with 0.993x
for uniform Host and 0.984x for uniform GPU.

**Fig. 9. Recovery latency and critical-agent latency.** Current data comes
from the implemented online SGLSim policy sweep
`sglsim-policy-sweep-agentic-r6-dp4-matched-seed-20260628`. Each plotted row
aggregates three matched-seed DP4 hard-restart trials and 72 replay rows. AegisKV
is 0.780x no-backup recompute on mean retry latency and 0.915x on
critical-agent p90 latency; uniform Host is 0.793x/0.908x and uniform GPUReplica
is 0.829x/0.935x.

**Fig. 10. Recovery path breakdown.** Current data comes from direct
multi-regime replay runtime counters and counterfactual path shares over the
same 264 hard-failure replay rows. The equal-regime AegisKV mix is 9.7% peer
GPU, 33.3% Host H2D, 0.0% ordinary cache, and 56.9% recompute. The appended
online rows now come from `sglsim-physical-recovery-action-agentic-r3-20260628`: DP2 Host
and DP4 peer long-prefix trials split mostly to Host restore with small
peer-GPU hot-tail shares, DP4 non-peer chooses Host restore, and DP4 peer
hot-tail chooses `gpu_replica` for all replay rows. Jobs `34357/34358/34359`
provide the recovery-shaped physical Host/GPUReplica restore anchor shown in
the plot annotation, while jobs `34338/34344/34345` remain the raw
H2D/D2H/P2P calibration. Final data still needs the live recovery path to
perform physical CUDA KV movement during end-to-end policy runs.

**Fig. 10b. Shared-prefix marginal accounting.** AegisKV avoids duplicated
protection of shared prefixes by scoring marginal safe-frontier coverage.
Current data: final SGLSim online shared-prefix sweep
`codex-shared-prefix-sweep-r3-20260628`, with nine runs, 141 multi-turn agent
groups, and observed shared-prefix bins spanning 2.5% to 77.5%.

**Fig. 11. Normal-path overhead.** AegisKV keeps no-failure overhead bounded by
sharing the fixed write-through Host path and limiting extra peer copies.
Current data: online SGLSim no-failure baseline/AegisKV sweep
`codex-normal-overhead-online-r3-20260628` gives -0.49% p99 overhead with
0.93% 95% CI and 1.28x write amplification, plus SGLang paired DP=2 smoke
provenance from jobs `32861`, `32863`, `32869`, `32870`, `32871`, `32874`,
and `32899` (mean overhead average 0.22%, p95 overhead average -0.19%, p99
overhead average 0.31%; all remain smoke-scale evidence).

**Fig. 11b. HostBackup visibility under fixed write-through.** Cross-rank
HostBackup visibility converts already-paid D2H copies into failure recovery
coverage. Current retry-latency data: SGLSim replay counterfactual over 192
recoverable Host/GPU rows; overhead axis remains a preliminary smoke/model
anchor. SGLang physical Host-restore parser rows now separately show hidden
frontier, full wait-complete materialization, and repeated partial best-effort
materialization: 11 parsed real-SGLang rows contain 7 materialized rows, 4 full
wait-complete materialized rows, 3 partial best-effort materialized rows, and
4 hidden-control rows. The current same-write best-effort aggregate is 0.961x
p99 with 0.016 95% CI. Final peer GPUReplica/P2P policy evidence is still
missing.

**Fig. 12. Ablation study.** Removing lifecycle awareness, segment-level
accounting, restore/recompute choice, pressure signals, hysteresis, or
critical-agent restore worsens WCT. Current data:
`sglsim-ablation-sweep-agentic-r4-dp4-hot-tail-matched-seed-20260628`, an
implemented online SGLSim ablation sweep with 7 variants, 21 trials, 0 failed
rows, and 504 replay rows. Strict-final evidence still requires live physical
KV movement rather than metadata materialization.

**Fig. 13. Budget sensitivity.** The best GPUReplica budget depends on what is
visible at retry time. Current data:
`sglsim-budget-sweep-agentic-r1-combined-20260628`, an implemented online DP4
hard-restart budget/timing sweep. Early decode improves from `1.000` to
`0.982`, mid decode to `0.986`, and late decode to `0.981` under the measured
r1 sweep. Strict-final evidence still needs live physical KV movement rather
than metadata materialization.

**Fig. 14. DP scale and peer availability.** The current direct SGLSim sweep
runs hard-failover replay at DP=2, DP=4, and DP=8. It shows stable completion
and cross-rank replay across larger DP topologies under a low-concurrency
workload; high-concurrency scale benefit remains final evidence. Current data:
SGLSim runtime replay-derived direct DP sweep.

**Fig. 14b. Peer-route selection.** In an online DP=4 short-tail route sweep,
routing to the recorded peer rank enables GPUReplica and reduces p99 recovery
overhead relative to a non-peer recompute route. Current data:
`sglsim-agentic-online-peer-route-sweep`; final data still needs physical peer
KV movement.

**Fig. 15. SGLang/SGLSim calibration anchor.** Current data compare the real
SGLang HiCache write-through normal-path p99 overhead aggregate, now 0.31%
across seven matched DP=2 jobs, with the online SGLSim AegisKV no-failure
overhead anchor, now -0.49% with 0.93% 95% CI. The plot is an overhead anchor,
not a final ranking-preservation result.

**Fig. 15f. Real SGLang Host restore materialization.** Current data summarize
the structured `physical_restore_runs.csv` parser output from real SGLang
server logs and failure summaries. The 11 rows include 4 hidden-control rows,
3 partial best-effort materialization rows, and 4 full wait-complete
materialization rows. Materialized rows account for 39,191 retry cached tokens
after accept. This is real Host restore data-path evidence; it does not close
the P0 large-sweep live `physical_tensor_copy` gate or validate peer
GPUReplica/P2P policy recovery.

## 最终数据到位后的变化

Final data should not change the structure of this section. It should replace
the evidence tier:

```text
sglsim-preliminary -> final
```

and update the numerical claims in the captions and result narrative. If final
data contradicts any preliminary trend, the paper must follow the final
measurements and either revise the claim or demote the figure to a diagnostic.
