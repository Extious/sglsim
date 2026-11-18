# SGLang Small-Scale Experiments

Target:

```text
host: hkbugpumanode
user observed by SSH: 25482351
repo: /home/comp/25482351/sglang
branch observed: multi-level-backup
observed SHA: 360072905
Slurm: sbatch, srun, sinfo available
```

Observed partitions include short A100/H20/RTX4080 queues. The first usable
paper run should target `short` with DP=2 on A100 if available.

First verified environment probe:

- Summary: `env_probe_32851_summary.md`
- Slurm job: `32851`
- Node: `hkbugpusrv12`
- GPU class: `NVIDIA A100-SXM4-80GB`
- SGLang branch/SHA: `multi-level-backup` / `360072905d9373a33fb96360474c9e6e1682f0ce`

DP=2 smoke status:

- `dp2_smoke_32852_failure.md`: first launch used the wrong Python environment.
- `dp2_smoke_32853_failure.md`: second launch reached DP model load and KV
  allocation, then failed in the flashinfer CUDA graph path because the node
  libstdc++ lacks `GLIBCXX_3.4.26`.
- `dp2_smoke_32854_failure.md`: third launch reached server ready with
  triton/pytorch backends and CUDA graph disabled, then exposed the cluster
  proxy issue for localhost warmup/client requests.
- `dp2_smoke_32855_success.md`: fourth launch completed the first end-to-end
  DP=2 SGLang smoke run, with 8/8 `/v1/completions` requests successful.
- `dp2_overhead_pair_32859_confounded.md`: first paired overhead run completed
  both cases, but is not final evidence because baseline and HiCache used
  different piecewise CUDA graph behavior.
- `dp2_overhead_pair_32861_summary.md`: paired same-node DP=2 no-failure
  overhead smoke with matched CUDA graph controls. Baseline and HiCache
  write-through both completed 8/8 requests; HiCache write-through showed
  2.02% p99 overhead and 3.05% mean latency overhead. This is usable
  small-scale provenance for Fig. 11, but not final HPCA evidence.
- `dp2_overhead_pair_32863_summary.md`: larger paired same-node DP=2
  no-failure overhead smoke with 32 requests and 128 max tokens per case.
  Baseline and HiCache write-through both completed 32/32 requests; mean
  latency overhead was 1.18%, while p99 overhead was -3.93%, indicating the
  tail metric remains within small-sample noise.
- `dp2_overhead_pair_32869_32871_summary.md`: three additional 32-request /
  128-token paired repeats. Jobs `32869`, `32870`, and `32871` all completed
  baseline and HiCache write-through cases with 32/32 successful requests per
  case. Mean overhead ranged from -0.49% to 2.28%; p99 overhead ranged from
  -2.93% to 3.74%.
- `dp2_overhead_pair_32874_summary.md`: larger 64-request / 128-token paired
  repeat. Job `32874` completed baseline and HiCache write-through cases with
  64/64 successful requests per case. Mean overhead was -2.23%, p95 overhead
  was -2.92%, and p99 overhead was 0.80%.
- `dp2_overhead_pair_32899_summary.md`: larger 128-request / 128-token paired
  repeat. Job `32899` completed baseline and HiCache write-through cases with
  128/128 successful requests per case. Mean overhead was -1.93%, p95 overhead
  was 1.43%, and p99 overhead was -0.80%.
- `results/aggregate_overhead_pairs/`: current aggregate includes jobs `32861`
  `32863`, `32869`, `32870`, `32871`, `32874`, and `32899`, excludes confounded
  job `32859`, and reports mean overhead average 0.22% with range -2.23% to
  3.05%, p95 overhead average -0.19% with range -4.09% to 2.02%, and p99
  overhead average 0.31% with range -3.93% to 3.74%.
- `dp2_failure_smoke_32977_summary.md`: DP=2 real SGLang
  failure-control-path smoke. Job `32977` completed 4/4 routed in-flight
  completion requests after `/simulate_gpu_failure`, captured and re-dispatched
  4 requests, returned 4 responses with `sglext.failover`, and recovered the
  failed rank. This is control-path evidence only; it does not yet measure
  Host-visible restore or peer GPUReplica physical KV-copy benefit.
- `dp2_failure_smoke_32978_summary.md`: longer DP=2 failure-control-path
  smoke with 512 max tokens and 8 s failure delay. Job `32978` again completed
  4/4 routed in-flight requests and returned 4 `sglext.failover` responses.
  It captured 760 pre-failover output tokens across the 4 requests, but still
  reported 0 pre-failover backed-up tokens. This is useful negative evidence:
  the current SGLang smoke path validates retry continuity but not an ACKed
  HostBackup frontier.
- `dp2_failure_smoke_32980_remote_backup_summary.md`: storage-backed DP=2
  failure smoke. Job `32980` starts a centralized `remote_backup` server and
  launches SGLang with `--hicache-storage-backend remote_backup`. It completed
  4/4 routed in-flight requests, returned 4 `sglext.failover` responses, and
  reported 5,132 pre-failover backed-up tokens. This is the first small-scale
  SGLang Host-visible frontier smoke in this artifact set.
- `dp2_failure_visibility_pair_32978_32980_summary.md`: paired posthoc
  Host-invisible vs remote-backup frontier summary for Fig. 11b/Fig. 15
  evidence. Job `32978` reports 0 backed-up tokens without a storage backend;
  job `32980` reports 5,132 backed-up tokens with `remote_backup`, or 1,283
  backed-up tokens/request. The paired p99 latency ratio is 1.108x, so this
  proves frontier visibility, not final restore-latency benefit.
- `dp2_failure_smoke_32981_request_aware_off_summary.md`: diagnostic rerun
  with `REMOTE_BACKUP_REQUEST_AWARE=0`. It still reports 5,132 backed-up
  tokens and 4/4 failover responses, but retry prefetch remains revoked with
  `data_pages=0`. This rules out request-aware lease GC as the main cause.
- `dp2_failure_smoke_32982_single_request_storage_hit_summary.md`:
  single-request remote-backup probe. It reports 1,297 backed-up tokens, DP1
  `data_pages=1297`, `Prefetch ACCEPTED`, and `prefetch_get` hits. The
  remaining gap is that `Prefetch IO END` reports `completed=0` and retry
  prefill still has `#cached-token: 0`.
- `dp2_failure_smoke_32983_wait_complete_restore_summary.md`:
  single-request remote-backup restore probe with
  `HICACHE_STORAGE_PREFETCH_POLICY=wait_complete`. It keeps the same
  page-0 shape as job `32982`, reports 1,297 backed-up tokens, completes
  prefetch IO with `completed=1297 / expected=1297`, and retries with
  `#cached-token: 1297`. This resolves the 32982 materialization gap for the
  single-request smoke and shows that the previous `completed=0` behavior was
  a prefetch termination policy issue.
- `dp2_failure_visibility_pair_32984_32983_summary.md`: single-request
  Host-invisible vs Host-visible pair. Job `32984` is a no-storage control
  with 0 backed-up tokens and 15.58 s p99 latency. Job `32983` is the
  materialized remote-backup restore smoke with 1,297 backed-up tokens and
  17.51 s p99 latency. The visible p99 ratio is 1.124x, so this is a
  correctness/calibration artifact rather than a final latency-benefit result.
- `dp2_failure_smoke_32985_wait_complete_four_request_summary.md`: four-request
  remote-backup probe with `wait_complete`. It again reports 5,132 backed-up
  tokens and 4/4 failover responses, but retry storage matching still reports
  `data_pages=0` for all four retried requests. This confirms that
  `wait_complete` fixes the single-request materialization race but not the
  concurrent split-prefix/storage-layout issue.
- `dp2_failure_smoke_32987_unique_prefix_wait_complete_summary.md`: four-request
  remote-backup restore probe with `UNIQUE_REQUEST_PREFIX=1` and
  `wait_complete`. It reports 5,204 backed-up tokens, `data_pages=1301` for
  all four retried requests, `completed=1301 / expected=1301`, and retry
  `#cached-token: 1301` for each request. This shows that concurrency itself
  is not the restore blocker when the backed data forms a page-0 prefix.
- `dp2_failure_visibility_pair_32989_32987_summary.md`: four-request
  unique-prefix Host-invisible vs Host-visible pair. Job `32989` is the
  no-storage control with 0 backed-up tokens and 16.73 s p99 latency. Job
  `32987` is the materialized remote-backup restore smoke with 5,204 backed-up
  tokens and 21.88 s p99 latency. The visible p99 ratio is 1.308x, so this is
  a correctness/calibration artifact rather than a final latency-benefit
  result.
- `dp2_failure_smoke_32991_patched_shared_prefix_restore_summary.md`:
  four-request shared-prefix restore probe after the SGLang split-prefix
  storage fix. Job `32991` reports 5,188 backed-up tokens, `data_pages=1297`
  for all four retried requests, `completed=1297 / expected=1297`, and retry
  `#cached-token: 1297` for each request. This resolves the shared-prefix
  materialization gap exposed by job `32985`.
- `dp2_failure_smoke_32992_patched_shared_prefix_no_storage_summary.md`:
  same-code shared-prefix no-storage control for job `32991`. Job `32992`
  reports 0 backed-up tokens, 4/4 failover responses, and 16.93 s p99 latency.
- `dp2_failure_visibility_pair_32992_32991_summary.md`: four-request
  shared-prefix Host-invisible vs Host-visible pair using the same patched
  SGLang working tree. Job `32992` is the no-storage control with 0 backed-up
  tokens and 16.93 s p99 latency. Job `32991` is the patched materialized
  remote-backup restore smoke with 5,188 backed-up tokens and 20.47 s p99
  latency. The visible p99 ratio is 1.210x, so this is the strongest current
  SGLang correctness/calibration artifact for shared-prefix restore, not a
  final latency-benefit result.
- `dp2_failure_visibility_pair_32993_32994_summary.md`: longer shared-prefix
  Host-invisible vs Host-visible pair with `PROMPT_REPEAT=256` and
  `FAILURE_DELAY_S=12`. Job `32993` is the no-storage control with 0 backed-up
  tokens and 22.06 s p99 latency. Job `32994` uses
  `remote_backup + wait_complete`, reports 20,548 backed-up tokens, and has
  35.96 s p99 latency. All four visible retry queries reach
  `data_pages=5137`, prefetch completes `5137 / 5137`, and retry prefill
  admits with `#cached-token: 5137`. The visible p99 ratio is 1.630x, so this
  extends the Host-frontier calibration range but still does not show final
  recovery-benefit.
- `dp2_failure_visibility_pair_32998_32999_summary.md`: same-write-cost
  visibility/isolation pair with `PROMPT_REPEAT=256` and `FAILURE_DELAY_S=12`.
  Both jobs enable `remote_backup + wait_complete` and report 20,548 backed-up
  tokens. Job `32998` sets `DISABLE_FAILOVER_REMOTE_LOOKUP=1`, so retry skips
  storage prefetch and only uses local/shared prefix hits. Job `32999` enables
  lookup and materializes `data_pages=5137` plus retry `#cached-token: 5137`
  for all four requests. The visible p99 ratio is 1.487x versus the
  lookup-disabled control, so this is a clean fixed-write-cost visibility
  artifact, not final recovery-benefit evidence.
- `dp2_failure_visibility_pair_33002_33003_summary.md`: same-write-cost
  `best_effort` visibility pair for the same long shared-prefix workload. Both
  jobs enable `remote_backup` and report 20,548 backed-up tokens. Job `33002`
  disables retry storage prefetch and has 26.43 s p99 latency. Job `33003`
  enables `best_effort` lookup: retry queries find `data_pages=5137`, prefetch
  is accepted for all four requests, one prefetch partially completes
  `2816 / 5137`, and retry prefill observes a partial `#cached-token: 2830`.
  Visible p99 is 25.52 s, a 3.4% reduction versus the lookup-disabled control.
  This is one member of the repeated same-write-cost best-effort aggregate
  below, and it is partial asynchronous restore evidence rather than full
  materialized restore evidence.
- `dp2_failure_visibility_pair_34333_34334_summary.md`: repeated same-write-cost
  `best_effort` pair. Both jobs again enable `remote_backup` writes; job
  `34333` disables retry lookup and job `34334` enables lookup. The visible
  path partially materializes 2,318 cached retry tokens and improves p99 to
  0.945x of the lookup-disabled control.
- `dp2_failure_visibility_pair_34336_34337_summary.md`: third repeated
  same-write-cost `best_effort` pair. Both jobs again enable `remote_backup`
  writes; job `34336` disables retry lookup and job `34337` enables lookup.
  The visible path partially materializes 1,806 cached retry tokens and improves
  p99 to 0.972x of the lookup-disabled control.
- `results/failure_visibility_pair_aggregate/`: aggregate over all paired
  visibility artifacts. The current same-write-cost `best_effort` aggregate
  combines `33002/33003`, `34333/34334`, and `34336/34337`, with
  visible/control p99 0.961x, 95% CI 0.016, and 3/3 improving pairs.
- `sglang_physical_restore_evidence_summary.md`: structured parser output over
  the current Host restore probes. It extracts `data_pages`, accepted prefetch
  pages, completed prefetch tokens, and retry `#cached-token` from real SGLang
  `server.log` files. The current CSV proves three physical Host-restore
  states: hidden frontier with no retry visibility, full prefetch
  materialization, and partial best-effort asynchronous materialization.
- `physical_kv_copy_microbench_aggregate_summary.md`: repeated physical CUDA
  movement calibration for AegisKV. Slurm jobs `34338`, `34344`, and `34345`
  ran on `hkbugpusrv14` with two A100 GPUs connected by `NV12`. With the
  Qwen3-8B bf16 KV-size assumption of 147,456 bytes/token, the 8192-token
  aggregate p50 latency is 48.595 ms H2D (95% CI 0.531 ms), 46.416 ms D2H
  (95% CI 0.100 ms), and 4.368 ms GPU0->GPU1 P2P (95% CI 0.004 ms). This
  calibrates physical copy costs for Fig. 2/Fig. 10, but remains a microbench,
  not end-to-end AegisKV policy evidence.
- `physical_recovery_path_microbench_aggregate_summary.md`: recovery-shaped
  physical movement calibration. Slurm jobs `34357`, `34358`, and `34359`
  reshape the raw CUDA copy into AegisKV backup/restore actions that write into
  retry-rank KV buffers. At 8192 tokens, Host restore H2D is 48.434 ms p50
  (95% CI 0.013 ms), GPUReplica restore P2P is 4.330 ms p50 (95% CI 0.005 ms),
  Host backup write D2H is 46.521 ms p50, and GPUReplica backup write P2P is
  4.324 ms p50. This is action-level physical movement evidence; final policy
  evidence still requires live recovery to invoke equivalent KV movement.
- The baseline smoke and paired-overhead Slurm scripts use `uv run python`,
  `--attention-backend triton`, `--sampling-backend pytorch`,
  `--disable-cuda-graph`, `--disable-piecewise-cuda-graph`, and localhost
  `NO_PROXY/no_proxy`.
- The failure smoke script defaults to `${SGLANG_DIR}/.venv/bin/python3`
  instead of launching server/client through `uv run`; this avoids losing GPU
  time to environment sync or lock waits during failure-path debugging.
- The failure smoke script excludes `hkbugpusrv07` and disables overlap
  scheduling by default. Job `32972` showed that this node's compiler runtime
  cannot build the SGLang JIT `resolve_future_token_ids` kernel because the C++
  `<version>` header is unavailable. The failure smoke is therefore scoped to
  control-path failover evidence rather than normal-path throughput evidence.
- The failure smoke script also prepends `${SGLANG_DIR}/.venv/bin` to `PATH`.
  Job `32976` reached `hkbugpusrv11` but crashed when JIT subprocess launch
  could not find an executable `ninja` through the inherited Slurm PATH.
- The failure smoke script can launch the remote backup server through
  `scripts/launch_remote_backup_server_for_smoke.py`, which exposes
  `REMOTE_BACKUP_REQUEST_AWARE`. The default is `1`, matching SGLang's
  request-aware lease behavior. Setting it to `0` is a diagnostic restore probe
  to test whether generation supersede/GC is responsible for
  `data_pages=0` prefetch misses after failover.
- The failure smoke script also exposes `HICACHE_STORAGE_PREFETCH_POLICY`.
  The default is SGLang's `best_effort`; setting it to `wait_complete` is a
  diagnostic restore probe that waits for remote-backup pages to materialize
  into the target-rank HiRadixCache before retry prefill admission.
- `UNIQUE_REQUEST_PREFIX=1` changes the diagnostic failure workload so each
  long prompt starts with a different first token. This intentionally removes
  shared-prefix radix splitting and is useful for proving that concurrent
  remote-backup restore works when data is backed from page 0.

## Purpose

SGLang supplies real-system evidence for:

- metadata and control-path overhead;
- fixed write-through D2H ACK latency;
- Host H2D restore latency;
- GPUReplica or peer-copy latency;
- failure retry timeline;
- Host-visible vs Host-invisible recovery under the same D2H cost;
- SGLang/SGLSim calibration points.

## Scripts

- `slurm/aegiskv_env_probe.sbatch`: records remote environment, GPU, Slurm, and repo state.
- `slurm/aegiskv_dp2_smoke.sbatch`: starts a DP=2 SGLang server with HiCache write-through and sends a small completion workload.
- `slurm/aegiskv_dp2_overhead_pair.sbatch`: runs paired DP=2 no-failure
  baseline vs HiCache write-through on the same Slurm allocation for Fig. 11
  normal-path overhead evidence.
- `slurm/aegiskv_dp2_failure_smoke.sbatch`: starts a DP=2 HiCache
  write-through SGLang server, routes concurrent completions to one DP rank,
  calls `/simulate_gpu_failure`, records `SGLANG_FAILOVER_EVENTS_FILE`, and
  checks that failover-retried responses carry `sglext.failover`.
- `scripts/sglang_smoke_client.py`: minimal HTTP client used by the smoke job.
- `scripts/sglang_failure_client.py`: concurrent failure-injection client used
  by the failure smoke job; emits request JSONL, summary JSON/CSV, and a
  preliminary candidate CSV.
- `scripts/summarize_smoke.py`: converts smoke JSONL output into preliminary CSVs.
- `scripts/summarize_overhead_pair.py`: aggregates paired baseline/HiCache
  latencies and emits a candidate Fig. 11 CSV.
- `scripts/summarize_failure_pair.py`: compares one Host-invisible failure
  smoke against one storage-backed failure smoke and emits the Fig. 11b/Fig. 15
  supporting paired frontier artifact.
- `scripts/extract_physical_restore_evidence.py`: parses real SGLang
  `server.log` files for Host restore materialization evidence (`data_pages`,
  prefetch completion, and retry `#cached-token`) and writes
  `results/physical_restore_evidence/physical_restore_runs.csv`.
- `scripts/plan_strict_final_recovery_matrix.py`: generates the current
  strict-final recovery-path Slurm matrix. It reuses the DP=2 failure smoke
  script with same-write-cost Host-invisible versus Host-visible pairs for
  `best_effort` and `wait_complete` restore policies, and emits submit,
  pair-sync, and physical-restore parser commands under
  `plans/strict_final_recovery_matrix/`.
- `scripts/audit_strict_final_recovery_matrix.py`: audits completed
  same-write-cost SGLang recovery pairs against that matrix. It accepts
  `best_effort` pairs only when recovery p99 improves and visible Host pages
  materialize, while `wait_complete` pairs are judged by full Host
  materialization rather than latency improvement.
- `scripts/prepare_strict_final_missing_jobs.py`: turns
  `strict_final_recovery_missing_jobs.csv` into executable submit/sync helper
  scripts. It emits both an SSH-alias submit script and a direct-proxy submit
  script that bypasses system `ssh_config` with `-F /dev/null`.
- `scripts/aggregate_failure_visibility_pairs.py`: aggregates paired SGLang
  failure-visibility summaries and writes
  `results/failure_visibility_pair_aggregate/`.
- `scripts/sync_failure_pair_from_remote.py`: given two Slurm job IDs, syncs
  the corresponding remote failure result directories and Slurm logs from
  `hkbugpumanode`, runs `summarize_failure_pair.py`, and writes a markdown
  pair summary.
- `scripts/aggregate_overhead_pairs.py`: scans all paired overhead result
  directories, excludes confounded runs whose server logs do not show matched
  controls, and emits an aggregate Fig. 11 candidate CSV.

These scripts are scaffolding. They verify that the real SGLang stack can run
and produce provenance. Full AegisKV figures still require the failure injection
and Host/GPUReplica metrics described in `../../paper_readiness_audit.md`.

## Sync and submit

From this repository root:

```bash
bash benchmark/AegisKV/presentation/experiments/sglang_small/scripts/sync_to_hkbugpumanode.sh
python benchmark/AegisKV/presentation/experiments/sglang_small/scripts/plan_strict_final_recovery_matrix.py
ssh hkbugpumanode 'cd ~/aegiskv_sglang_small && sbatch slurm/aegiskv_env_probe.sbatch'
ssh hkbugpumanode 'cd ~/aegiskv_sglang_small && sbatch slurm/aegiskv_dp2_smoke.sbatch'
ssh hkbugpumanode 'cd ~/aegiskv_sglang_small && sbatch slurm/aegiskv_dp2_overhead_pair.sbatch'
ssh hkbugpumanode 'cd ~/aegiskv_sglang_small && sbatch slurm/aegiskv_dp2_failure_smoke.sbatch'
ssh hkbugpumanode 'cd ~/aegiskv_sglang_small && ENABLE_REMOTE_BACKUP=1 NUM_REQUESTS=4 MAX_TOKENS=512 FAILURE_DELAY_S=8 REMOTE_BACKUP_BUFFER_GB=16 sbatch slurm/aegiskv_dp2_failure_smoke.sbatch'
ssh hkbugpumanode 'cd ~/aegiskv_sglang_small && ENABLE_REMOTE_BACKUP=1 REMOTE_BACKUP_REQUEST_AWARE=0 NUM_REQUESTS=4 MAX_TOKENS=512 FAILURE_DELAY_S=8 REMOTE_BACKUP_BUFFER_GB=16 sbatch slurm/aegiskv_dp2_failure_smoke.sbatch'
ssh hkbugpumanode 'cd ~/aegiskv_sglang_small && ENABLE_REMOTE_BACKUP=1 HICACHE_STORAGE_PREFETCH_POLICY=wait_complete NUM_REQUESTS=1 MAX_TOKENS=512 FAILURE_DELAY_S=8 REMOTE_BACKUP_BUFFER_GB=16 sbatch slurm/aegiskv_dp2_failure_smoke.sbatch'
ssh hkbugpumanode 'cd ~/aegiskv_sglang_small && ENABLE_REMOTE_BACKUP=1 HICACHE_STORAGE_PREFETCH_POLICY=wait_complete UNIQUE_REQUEST_PREFIX=1 NUM_REQUESTS=4 MAX_TOKENS=512 FAILURE_DELAY_S=8 REMOTE_BACKUP_BUFFER_GB=16 sbatch slurm/aegiskv_dp2_failure_smoke.sbatch'
ssh hkbugpumanode 'cd ~/aegiskv_sglang_small && ENABLE_REMOTE_BACKUP=0 UNIQUE_REQUEST_PREFIX=1 NUM_REQUESTS=4 MAX_TOKENS=512 FAILURE_DELAY_S=8 sbatch slurm/aegiskv_dp2_failure_smoke.sbatch'
ssh hkbugpumanode 'cd ~/aegiskv_sglang_small && ENABLE_REMOTE_BACKUP=1 HICACHE_STORAGE_PREFETCH_POLICY=wait_complete UNIQUE_REQUEST_PREFIX=0 NUM_REQUESTS=4 MAX_TOKENS=512 FAILURE_DELAY_S=8 REMOTE_BACKUP_BUFFER_GB=16 sbatch slurm/aegiskv_dp2_failure_smoke.sbatch'
ssh hkbugpumanode 'cd ~/aegiskv_sglang_small && ENABLE_REMOTE_BACKUP=0 UNIQUE_REQUEST_PREFIX=0 NUM_REQUESTS=4 MAX_TOKENS=512 FAILURE_DELAY_S=8 sbatch slurm/aegiskv_dp2_failure_smoke.sbatch'
```

For the strict-final recovery blocker, prefer the generated matrix over the
older ad-hoc commands above:

```text
benchmark/AegisKV/presentation/experiments/sglang_small/plans/strict_final_recovery_matrix/
```

The default matrix contains three repeats of:

- `same_write_best_effort`: both runs enable remote_backup writes; the control
  disables failover lookup while the visible run allows Host lookup. This is the
  recovery-latency benefit candidate.
- `same_write_wait_complete`: both runs enable remote_backup writes and use
  wait-complete restore. This is the full-materialization correctness candidate.

The generated plan is not final evidence by itself. It is the executable path to
collect real SGLang recovery data-plane evidence without relabeling SGLSim
metadata-materialization rows as physical tensor-copy rows.

After the planned jobs have been synced and parsed, run:

```bash
python benchmark/AegisKV/presentation/experiments/sglang_small/scripts/audit_strict_final_recovery_matrix.py
```

The audit writes:

```text
benchmark/AegisKV/presentation/experiments/sglang_small/results/strict_final_recovery_matrix_audit/
```

The same directory also contains `strict_final_recovery_missing_jobs.csv`,
which is generated from the current accepted-pair counts and the matrix plan.
At the current snapshot it selects the two missing wait-complete repeats
(`strict_final_same_write_wait_complete_r2` and
`strict_final_same_write_wait_complete_r3`). A `pass` in this report means the
Host recovery-path matrix is complete; it still does not prove peer GPUReplica
P2P recovery or make SGLSim P0 restore rows strict-final without an explicit
final evidence connection.

To materialize executable helpers for the missing jobs:

```bash
python benchmark/AegisKV/presentation/experiments/sglang_small/scripts/prepare_strict_final_missing_jobs.py
```

This writes the following files under the same audit directory:

```text
submit_missing_alias.sh
submit_missing_direct_proxy.sh
sync_missing_template.sh
strict_final_recovery_submission_manifest.{json,md}
```

Use `submit_missing_alias.sh` when the `hkbugpumanode` SSH alias works. Use
`submit_missing_direct_proxy.sh` when the local system `ssh_config` is broken
or the alias cannot be resolved; it explicitly uses `25482351@172.27.244.19`
through jump host `zhanzhao@158.182.7.1`.

Recommended execution pattern:

```bash
AUDIT_DIR=benchmark/AegisKV/presentation/experiments/sglang_small/results/strict_final_recovery_matrix_audit
"${AUDIT_DIR}/submit_missing_direct_proxy.sh" 2>&1 | tee "${AUDIT_DIR}/submit_missing_direct_proxy.log"
python benchmark/AegisKV/presentation/experiments/sglang_small/scripts/bind_strict_final_job_ids.py \
  --submit-log "${AUDIT_DIR}/submit_missing_direct_proxy.log"
"${AUDIT_DIR}/sync_bound_jobs.sh"
python benchmark/AegisKV/presentation/experiments/sglang_small/scripts/audit_strict_final_recovery_matrix.py
```

`bind_strict_final_job_ids.py` parses `Submitted batch job <id>` lines from
the submit log and writes `strict_final_recovery_job_bindings.{csv,json,md}`.
If any job id is still missing, `sync_bound_jobs.sh` exits before running
network commands. Successful syncs write per-pair physical-restore parser
outputs under
`results/physical_restore_evidence/<pair_id>/physical_restore_runs.csv`; the
matrix audit discovers those files automatically together with the aggregate
`results/physical_restore_evidence/physical_restore_runs.csv`.

After a paired failure probe finishes, sync and summarize it from the local
repository:

```bash
python benchmark/AegisKV/presentation/experiments/sglang_small/scripts/sync_failure_pair_from_remote.py \
  --invisible-job-id <no_storage_job> \
  --visible-job-id <remote_backup_job> \
  --run-id failure_visibility_pair_<no_storage_job>_<remote_backup_job>
```

After completion, copy results back into:

```text
benchmark/AegisKV/presentation/experiments/sglang_small/results/
```

Then use measured aggregate CSVs to populate:

```text
benchmark/AegisKV/presentation/figures/data/measured/
```

To aggregate completed paired overhead jobs without promoting them to final
evidence:

```bash
python benchmark/AegisKV/presentation/experiments/sglang_small/scripts/aggregate_overhead_pairs.py
```

The aggregate output is written under:

```text
benchmark/AegisKV/presentation/experiments/sglang_small/results/aggregate_overhead_pairs/
```

To rebuild the 32978/32980 failure-visibility paired artifact:

```bash
python benchmark/AegisKV/presentation/experiments/sglang_small/scripts/summarize_failure_pair.py \
  --invisible-dir benchmark/AegisKV/presentation/experiments/sglang_small/results/dp2_failure_smoke_32978 \
  --visible-dir benchmark/AegisKV/presentation/experiments/sglang_small/results/dp2_failure_smoke_32980 \
  --output-dir benchmark/AegisKV/presentation/experiments/sglang_small/results/failure_visibility_pair_32978_32980 \
  --run-id sglang-dp2-failure-visibility-pair-32978-32980 \
  --invisible-job-id 32978 \
  --visible-job-id 32980
```
