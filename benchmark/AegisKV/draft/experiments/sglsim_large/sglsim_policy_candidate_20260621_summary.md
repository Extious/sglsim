# SGLSim Policy-Candidate Derivation: 2026-06-21

Status: schema-valid SGLSim serving-derived candidate data, not final paper
evidence.

## Run

- Script: `scripts/run_policy_candidate_smoke.sh`
- Config: `config_policy_candidate/`
- Run directory: `results/policy_candidate_synthesis/20260621_025456`
- Derived candidate directory:
  `results/derived_policy_candidates/sglsim-policy-candidate-20260621_025456/candidate_measured/`
- Model: `mock-model`
- Parallelism: `dp_size=2`, `tp_size=1`
- Workload: 32 generated shared-prefix agents, 3 turns each
- Critical-agent ratio: 25%
- Completed request rows: 96
- Failed request rows: 0

## Online AegisKV Metric Run

After adding runtime-emitted AegisKV bookkeeping fields to `sim_stats` and
`request_details.csv`, the same candidate workload was rerun:

- Run directory: `results/policy_candidate_synthesis/20260621_030638`
- Derived candidate directory:
  `results/derived_policy_candidates/sglsim-policy-candidate-20260621_030638/candidate_measured/`
- Completed request rows: 96
- Failed request rows: 0
- `aegis_host_visible_rate`: 24.52%
- `aegis_gpu_replica_rate`: 3.06%
- `aegis_safe_frontier_rate`: 79.18%
- `aegis_recompute_rate`: 20.82%
- mean `aegis_expected_recovery_ms`: 15.38 ms

The new request detail columns are:

```text
aegis_policy
aegis_backup_kv
aegis_worker_seq
aegis_assigned_dp_rank
aegis_host_visible_tokens
aegis_gpu_replica_tokens
aegis_safe_frontier_tokens
aegis_recompute_tokens
aegis_recovery_action
aegis_expected_recovery_ms
```

## Online Candidate Failover Fields

After adding online what-if failover fields and aligning recovery estimates to
the same restore/recompute lower-envelope, the candidate workload was rerun
again:

- Run directory: `results/policy_candidate_synthesis/20260621_032719`
- Derived candidate directory:
  `results/derived_policy_candidates/sglsim-policy-candidate-20260621_032719/candidate_measured/`
- Completed request rows: 96
- Failed request rows: 0
- Candidate retry rows: 24
- Candidate backed-up token rate at failure point: 100.00%
- Candidate recompute token rate at failure point: 0.00%
- Candidate recovery path: 100.00% Host restore, 0.00% peer GPU,
  0.00% ordinary cache
- Mean candidate failover overhead: 13.45 ms
- Mean `aegis_expected_recovery_ms` after lower-envelope alignment: 20.36 ms

The additional request detail columns are:

```text
aegis_candidate_retry
aegis_candidate_failed_dp_rank
aegis_candidate_pre_failure_output_tokens
aegis_candidate_pre_failure_total_tokens
aegis_candidate_backed_up_tokens
aegis_candidate_recompute_tokens
aegis_candidate_recovery_action
aegis_candidate_failover_overhead_s
```

These fields are still candidate recovery events: they are computed from the
online request's observed Aegis frontier and do not abort or retry the live
request.

## Failover Replay Smoke

After adding optional benchmark-level failover replay, scheduler subprocess
kill/restart control-plane injection, and runtime HostBackup / peer GPUReplica
directory lookup, the same workload shape was run with
`gsp_failover_replay=true` and `gsp_failover_injection=hard-restart` for
critical agents:

- Run directory: `results/failover_replay_synthesis/20260621_042727`
- Completed request rows: 120
- Failed request rows: 0
- Original turn rows: 96
- HTTP retry replay rows: 24
- Rows marked `aegis_failover_replay=true`: 24
- Rows marked `aegis_cross_rank_retry=true`: 24
- Rows marked `aegis_hard_failure_injected=true`: 24
- Rows marked `aegis_host_backup_lookup_hit=true`: 24
- Rows marked `aegis_peer_gpu_lookup_hit=true`: 24
- HostBackup directory lookup tokens: 39,676
- peer GPUReplica directory lookup tokens: 5,036
- Candidate retry rows, including replay rows: 48
- Candidate recovery path: 100.00% Host restore
- Replay restored tokens: 39,676
- Replay restored prefill tokens materialized as target-rank cache: 38,989
- Replay recompute tokens: 0
- Mean candidate failover overhead: 13.45 ms
- Mean observed replay restore sleep: 13.51 ms

This run sends real retry-shaped HTTP requests through the live
tokenizer/scheduler/detokenizer path after protected critical turns, calls
`/simulate/kill_dp` before each replay request, and forces those replay requests
to a different DP rank from the recorded failed source rank. After the replay,
the client calls `/simulate/restart_dp` to bring the killed scheduler rank back.
The Engine parent resolves the restore plan from its runtime HostBackup and
peer GPUReplica directories before dispatching the replay row. In this
long-prefix run, peer entries are present for all replay rows, but the
lower-envelope chooses Host restore because the peer hot-tail replica would
require recomputing most of the prefix.
The scheduler sleeps for the restore plan before admitting each replay row, so
restore overhead is visible in wall-clock and queue-time metrics. The scheduler
also materializes restored prompt-prefix KV as target-rank device-cache metadata
before replay admission. Real CUDA tensor KV copy remains outside this
pure-metadata simulator.

## Observed SGLSim Serving Metrics

The table below reflects the latest failover replay run
`results/failover_replay_synthesis/20260621_042727`, whose derived candidate
CSVs are the current inputs copied into `figures/data/measured/`.

| Metric | Value |
|---|---:|
| total input tokens | 198127 |
| total output tokens | 6774 |
| mean E2E latency | 103.97 ms |
| p90 E2E latency | 121.85 ms |
| max / p99 proxy E2E latency | 135.52 / 134.04 ms |
| mean input tokens per request row | 1651.06 |
| p90 input tokens per request row | 2176 |
| cache-hit token rate | 66.81% |
| derived shared-prefix ratio | 66.81% |
| critical-agent row ratio | 40.00% |

## Candidate Figure Coverage

The derivation script emitted schema-valid candidate CSVs for all 14
quantitative result figures:

```text
fig_motivation_equal_size_loss
fig_restore_recompute_boundary
fig_static_gpu_replica_budget_gap
fig_overall_wct
fig_recovery_latency
fig_recovery_path_breakdown
fig_shared_prefix_marginal
fig_normal_path_overhead
fig_hostbackup_visibility
fig_ablation
fig_budget_sensitivity
fig_dp_scale
fig_peer_rank_selection
fig_simulator_calibration
```

`fig_normal_path_overhead.csv` is now intentionally mixed preliminary data:
four rows come from the SGLSim policy-candidate model and two rows come from
the matched SGLang DP=2 paired aggregate (`32861+32863`) on `hkbugpumanode`.
This makes the normal-path overhead figure better grounded, while still keeping
the evidence tier below final.

## Interpretation

This is a useful step beyond hand-written expected trends because the policy
model is anchored to a real SGLSim HTTP serving run and uses observed
`request_details.csv` fields such as input length, output length, cache hits,
load-back tokens, per-request latency, agent id, critical-agent metadata, and
runtime-emitted AegisKV frontier/protection fields.

It is still not final HPCA evidence. The AegisKV placement, failure injection,
safe-frontier publication, Host-visible retry path, and GPUReplica peer
restore are not yet executed as full online failure-recovery paths. Candidate
CSVs may be copied into `figures/data/measured/*.csv` only with a non-final
SGLSim evidence tier; they must not be marked `evidence_tier=final`.
