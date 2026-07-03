# SGLSim Large-Scale Experiments

SGLSim is the source for scale and sensitivity figures. The visible worktree now
has a maintainable `benchmark/agentic_workflows/` wrapper for named agentic
workflow scenarios, but still lacks the final online AegisKV policy path with
physical KV movement.

The currently maintainable online benchmark entry is:

```text
benchmark/synthesis_requests/run.py
```

The agentic workflow wrapper is:

```text
benchmark/agentic_workflows/config.py
benchmark/agentic_workflows/run.py
benchmark/agentic_workflows/suite.py
benchmark/agentic_workflows/summary.py
benchmark/agentic_workflows/combine.py
benchmark/agentic_workflows/paper.py
benchmark/agentic_workflows/aggregate.py
```

It reuses `benchmark/synthesis_requests/run.py` for every real run, generates
repeatable per-scenario config directories, summarizes `request_details.csv`,
and exports paper-facing candidate rows with
`evidence_tier=sglsim-agentic-online-preliminary`. It is an execution scaffold,
not final evidence.

Dedicated Fig. 10b shared-prefix sweep:

```text
scripts/run_shared_prefix_sweep.py
```

This entry runs `shared_prefix_sweep_low`, `shared_prefix_sweep_mid`, and
`shared_prefix_sweep_high` agentic scenarios, then derives
`fig_shared_prefix_marginal.csv` from original multi-turn request rows. The
three scenarios vary both turn count and shared-prefix/question length so the
resulting bins cover low, medium, and high shared-content overlap instead of
only the narrow three-turn range used by the replay matrices. The script
checks the observed shared-prefix span before writing the CSV. It writes
preliminary rows by default; `--final-evidence` should only be used after the
sweep satisfies the final evidence matrix requirement.

The AegisKV experiment-facing suite entry is:

```text
scripts/run_agentic_workflow_suite.py
```

It runs the wrapper from this experiment directory and writes combined summary
and paper-candidate CSVs under `results/agentic_workflows/`.

Staged 1-3-1-3-1 motivation and alpha figures can be refreshed from the current
stage-barrier trace and online adaptive-alpha sweep with:

```text
scripts/refresh_staged_13131_figures.py
```

By default it rebuilds:

- `fig_motivation_equal_size_loss.csv`
- `fig_critical_backup_priority.csv`
- `fig_critical_agent_profile.csv`
- `fig_alpha_workload_failure_policy.csv`
- `fig_online_alpha_adaptation.csv`
- `fig_staged_alpha_risk_tradeoff.csv`

The online alpha figure is backed by
`results/agentic_workflows/sglsim-staged-13131-alpha-online-sweep-20260702`,
which runs low/high workload-pressure points and multiple failure-rate states.

The Beamer deck intentionally uses only the generated System overview plus
quantitative result PNGs. It does not insert detailed draw.io mechanism
flowcharts such as backup data path, decision loop, or failure timeline.

`config_smoke/` and `scripts/run_synthesis_smoke.sh` validate that this entry
can run from the AegisKV experiment area and export a smoke summary. This smoke
path is not AegisKV paper evidence; it only verifies the local SGLSim plumbing.

`config_policy_candidate/` and `scripts/run_policy_candidate_smoke.sh` run a
larger mock-model workflow workload and then derive per-figure candidate CSVs
from the resulting `request_details.csv`. These candidates are labeled
`evidence_tier=sglsim-synthesis-derived`; they are stronger than expected-trend
templates because they are anchored to an actual SGLSim serving run, but they
still are not final online AegisKV policy evidence.

`config_failover_replay/` and `scripts/run_failover_replay_smoke.sh` enable the
optional synthesis client replay path. For protected critical turns, the client
sends an additional retry-shaped HTTP request marked
`aegis_failover_replay=true`. The current smoke uses
`gsp_failover_injection=hard-restart`: before each replay, the client terminates
the recorded source scheduler subprocess through `/simulate/kill_dp`, forces the
retry to a different DP rank, then restarts the source rank through
`/simulate/restart_dp`. The Engine parent fills the replay restore plan from its
runtime HostBackup and peer GPUReplica directories, so replay rows no longer
carry client-provided restore-token counts. It remains a benchmark-level replay
smoke rather than a physical KV page-copy protocol, but the failure is now a
real scheduler subprocess death and the restore decision comes from runtime
directory lookup.

`scripts/generate_counterfactual_results.py` generates deterministic
`evidence_tier=sglsim-preliminary` CSVs for all quantitative figures. These
CSVs replace templates for internal paper review, but they are not final
evidence.

First local smoke summary:

- `sglsim_smoke_20260621_summary.md`
- run directory: `results/synthesis_smoke/20260621_021131`
- completed requests: `8`
- failed requests: `0`

First policy-candidate derivation:

- run id: `sglsim-derived-smoke-20260621`
- candidate directory:
  `results/derived_policy_candidates/sglsim-derived-smoke-20260621/candidate_measured/`
- completed request rows: `8`
- failed request rows: `0`
- status: schema-valid candidate CSVs for all 14 quantitative figures.

First larger policy-candidate serving run:

- Summary: `sglsim_policy_candidate_20260621_summary.md`
- run id: `sglsim-policy-candidate-20260621_025456`
- run directory: `results/policy_candidate_synthesis/20260621_025456`
- candidate directory:
  `results/derived_policy_candidates/sglsim-policy-candidate-20260621_025456/candidate_measured/`
- completed request rows: `96`
- failed request rows: `0`
- observed cache-hit token rate: `74.60%`
- critical-agent row ratio: `25.00%`

Online AegisKV metric validation:

- run id: `sglsim-policy-candidate-20260621_030638`
- run directory: `results/policy_candidate_synthesis/20260621_030638`
- completed request rows: `96`
- failed request rows: `0`
- `aegis_safe_frontier_rate`: `79.18%`
- `aegis_host_visible_rate`: `24.52%`
- `aegis_gpu_replica_rate`: `3.06%`
- `aegis_recompute_rate`: `20.82%`

Online failover replay smoke:

- run directory: `results/failover_replay_synthesis/20260621_042727`
- completed request rows: `120`
- failed request rows: `0`
- HTTP retry replay rows: `24`
- rows marked `aegis_failover_replay=true`: `24`
- rows marked `aegis_cross_rank_retry=true`: `24`
- rows marked `aegis_hard_failure_injected=true`: `24`
- rows marked `aegis_host_backup_lookup_hit=true`: `24`
- rows marked `aegis_peer_gpu_lookup_hit=true`: `24`
- HostBackup lookup tokens: `39676`
- peer GPUReplica lookup tokens: `5036`
- replay restored tokens: `39676`
- replay restored prefill tokens materialized as target-rank cache: `38989`
- replay recompute tokens: `0`
- mean observed replay restore sleep: `13.51 ms`

Repeated failover replay matrix:

- runner: `scripts/run_failover_replay_matrix.py`
- current run id: `sglsim-failover-replay-matrix-r5-20260621`
- completed seeds: `5`
- completed request rows: `600`
- failed request rows: `0`
- HTTP retry replay rows: `120`
- cross-rank hard-failure replay rows: `120`
- HostBackup lookup hits: `120`
- peer GPUReplica lookup hits: `120`
- HostBackup lookup tokens: `201787`
- replay restored prefill tokens materialized as target-rank cache: `198322`
- replay recompute tokens: `0`
- mean observed replay restore sleep: `13.62 ms`
- candidate CSV directory:
  `results/failover_replay_matrix/sglsim-failover-replay-matrix-r5-20260621/derived_policy_candidates/candidate_measured/`
- current figure CSV aggregation: mean/std/95% CI over the 5 run-level
  SGLSim candidate derivations, plus the seven-job SGLang DP=2 Fig. 11 overhead
  aggregate for `sglang-small-paired` rows.
- Fig. 8/9 staged online policy/recovery sweep:
  `scripts/run_agentic_workflow_suite.py` plus
  `scripts/derive_agentic_policy_matrix_from_suite.py` and
  `scripts/derive_agentic_recovery_latency_from_suite.py`
- Fig. 8/9 run id:
  `sglsim-staged-13131-policy-stress-v2-r3-20260702`
- Fig. 8/9 workload:
  online SGLSim 1-3-1-3-1 stage-barrier workflow, 27 agent requests per
  trial, 3 matched seeds per policy, DP4 soft-failure replay, Critical-only
  backup/retry actions, output length 128 with failure after 128 output tokens,
  and physical-copy-calibrated recovery timing.
- Fig. 8/9 result: Fig. 8 now computes both workflow-level total WCT and
  failure-exposed restore cost directly from the same `request_details.csv`
  files, rather than per-request mean latency. It reports normalized total WCT
  of 1.000 no backup, 0.982 uniform Host, 0.987 uniform GPU, and 0.993
  AegisKV; WCT differences remain small because normal serving time still
  dominates the workflow. The failure-exposed restore cost is 1.000 no backup,
  0.760 uniform Host, 0.742 uniform GPU, and 0.699 AegisKV. Fig. 9 reports the
  same retry-latency ratios and shows the AegisKV policy chose 72 Host restores
  and 63 peer-GPU restores across 135 replay actions. The live movement audit covers
  540 replay rows and 405 restore rows; these rows are still
  `metadata_materialization` with `physical_recovery_path_csv` timing, not
  strict-final physical tensor movement.
- direct Fig. 2 restore/recompute boundary derivation:
  `scripts/derive_restore_boundary_from_replay.py`
- direct Fig. 2 run id:
  `sglsim-fig2-restore-boundary-calibrated-r11-20260621`
- direct Fig. 2 result: 264 hard-failure replay rows fit recompute
  `4.0 + 0.0108 * tokens` ms, Host `7.5 + 0.0036 * tokens` ms, and
  GPUReplica `2.4 + 0.0019 * peer_tokens + 0.0108 * suffix_tokens` ms, with
  34.2% observed peer-token coverage in the GPUReplica regime.
Direct DP-scale sweep for Fig. 14:

- runner: `scripts/run_failover_replay_matrix.py` plus
  `scripts/derive_dp_scale_from_summaries.py`
- run id: `sglsim-direct-dp-scale-dp2r5-dp4r3-dp8r3-20260621`
- source runs: DP=2 r5 matrix, DP=4 r3 matrix, DP=8 r3 matrix
- completed request rows: `1320`
- failed request rows: `0`
- hard-failure replay rows: `264`
- candidate CSV:
  `results/failover_replay_matrix/direct_dp_scale_20260621/fig_dp_scale.csv`
- current interpretation: direct serving-path DP-size smoke; still below final
  high-concurrency AegisKV scale evidence.

Direct peer-route replay pair for Fig. 14b:

- runner: `scripts/run_failover_replay_matrix.py` plus
  `scripts/derive_peer_route_selection_from_replay.py`
- run id: `sglsim-fig14b-peer-route-dp4-r3-20260621`
- source runs:
  `sglsim-fig14b-peer-route-dp4-peer-r3-20260621` and
  `sglsim-fig14b-peer-route-dp4-nonpeer-r3-20260621`
- workload: DP=4 short-tail hard-failover replay, identical seeds and request
  IDs, retry offset 1 versus offset 2
- matched recovery rows: `72`
- route outcomes: peer route selects `gpu_replica` on 72/72 recovery rows;
  non-peer route selects `recompute` on 72/72 recovery rows
- current result: AegisKV peer route is `0.978x` normalized WCT and `70.7%`
  p99 recovery overhead relative to the non-peer recompute route
- current interpretation: direct route-hit proxy; still below final physical
  peer KV-copy policy evidence.

Recovery-action-calibrated agentic suite for Fig. 10/Fig. 14:

- wrapper: `scripts/run_agentic_workflow_suite.py`
- run id: `sglsim-current-physical-recovery-action-agentic-r3-20260702`
- environment:
  `SGLSIM_AEGIS_RECOVERY_MODEL=physical_copy` and
  `SGLSIM_AEGIS_PHYSICAL_COPY_CSV=.../physical_recovery_path_aggregate.csv`
- scenarios: DP2 Host, DP4 peer, DP4 non-peer, DP4 peer hot-tail
- repeats: 3 per scenario
- completed request rows: `1440`
- failed request rows: `0`
- hard-failure replay rows: `288`
- mean restore sleep: `17.149 ms` DP2 Host, `16.914 ms` DP4 peer,
  `17.342 ms` DP4 non-peer, and `5.009 ms` DP4 peer hot-tail
- live movement audit: `288/288` restore rows report
  `metadata_materialization` with timing source `physical_recovery_path_csv`;
  `physical_tensor_copy` rows remain `0`.
- current result: Fig. 10 and Fig. 14 online agentic rows now use this
  recovery-action-calibrated timing suite.
- current interpretation: stronger preliminary timing evidence; still metadata
  materialization rather than physical CUDA tensor KV movement.

Live physical-movement audit:

- script: `scripts/audit_live_physical_movement.py`
- current smoke runner: `scripts/run_current_physical_recovery_smoke.py`
- current smoke run id: `sglsim-current-physical-recovery-smoke-20260702`
- current smoke command shape:

```bash
python benchmark/AegisKV/draft/experiments/sglsim_large/scripts/run_current_physical_recovery_smoke.py \
  --run-id sglsim-current-physical-recovery-smoke-20260702 \
  --runs 1
```

- current smoke result: `8/8` requests completed, `4` hard-failover replay
  restore rows, `106` restored tokens, and `4.466 ms` mean restore sleep.
- current audit result: strict audit fails as expected because 4/4 restore
  rows use `aegis_replay_restore_data_plane=metadata_materialization`; their
  timing source is `physical_recovery_path_csv`.
- audit artifacts:
  `results/agentic_workflows/sglsim-current-physical-recovery-smoke-20260702/live_physical_movement_audit.csv`
  and
  `results/agentic_workflows/sglsim-current-physical-recovery-smoke-20260702/live_physical_movement_audit.md`
- direct audit command shape:

```bash
python benchmark/AegisKV/draft/experiments/sglsim_large/scripts/audit_live_physical_movement.py \
  --suite-root benchmark/AegisKV/draft/experiments/sglsim_large/results/agentic_workflows/sglsim-current-physical-recovery-smoke-20260702 \
  --strict
```

- final gate: replay restore rows must use
  `aegis_replay_restore_data_plane=physical_tensor_copy`.

Agentic workflow wrapper validation:

- wrapper: `benchmark/agentic_workflows/`
- experiment entry: `scripts/run_agentic_workflow_suite.py`
- smoke run id: `codex-smoke-20260621`
- smoke result: `4/4` turn rows completed with 0 failures.
- hard-failover HostBackup run id: `codex-hard-host-20260621`
- hard-failover result: `120/120` rows completed with 0 failures, including
  `24` hard-failure replay rows, `24` HostBackup lookup hits, `24` peer
  GPUReplica lookup hits, `38,868` restored HostBackup tokens, `38,154`
  restored prefill tokens materialized as target-rank metadata cache, and
  `0` replay recompute tokens.
- DP4 route suite id: `codex-peer-nonpeer-20260621`
- DP4 peer result: `120/120` rows completed, `24` hard-failure replay rows,
  `24` HostBackup lookup hits, `24` peer GPUReplica lookup hits, `38,047`
  restored tokens, and `0` replay recompute tokens.
- DP4 non-peer result: `120/120` rows completed, `24` hard-failure replay
  rows, `24` HostBackup lookup hits, `0` peer GPUReplica lookup hits,
  `39,275` restored tokens, and `0` replay recompute tokens.
- repeated online suite id: `sglsim-agentic-online-r3-20260621`
- repeated suite result: four named scenarios, 12 total trials, and
  `0` failed rows. The aggregate CSV reports `288/288` completed rows for
  DP2 shared-prefix normal path and `360/360` completed rows for each
  hard-failover scenario. DP2 HostBackup, DP4 peer, and DP4 non-peer each
  produced `72` hard replay rows and `72` HostBackup lookup hits; DP4 peer
  produced `72` peer GPUReplica lookup hits while DP4 non-peer produced `0`.
  Mean restored tokens are `39,791`, `39,023`, and `39,479` for HostBackup,
  peer, and non-peer respectively, all with `0` recompute tokens.
- exported candidate CSV:
  `benchmark/agentic_workflows/results/agentic_workflow_paper_candidates.csv`
- repeated-suite aggregate CSV:
  `results/agentic_workflows/sglsim-agentic-online-r3-20260621/agentic_workflow_paper_aggregate.csv`
- current interpretation: reproducible online workflow wrapper validation;
  selected rows are now consumed by Fig. 14 as
  `sglsim-agentic-online-preliminary`, not final measured figure data.

Before large-scale figures can become final:

1. Use `benchmark/agentic_workflows/` to run repeated named online workflow
   scenarios instead of ad-hoc per-figure scripts.
2. Add final online AegisKV policy integration and real tensor-copy validation.
   Aegis metadata, safe-frontier accounting, runtime HostBackup and peer
   GPUReplica directory lookup, restore-vs-recompute selection, retry
   generation, scheduler subprocess failure injection, and target-rank
   metadata page movement now exist as preliminary plumbing.
3. Export per-run CSVs with the schema in `../schema/figure_result_columns.md`.
4. Aggregate those runs into measured CSVs under `../../figures/data/measured/`.

The final replacement criterion is explicit: every quantitative CSV must use
`evidence_tier=final` before `validate_figure_evidence.py --strict-final` can
pass.

## Smoke

From the repository root:

```bash
bash benchmark/AegisKV/draft/experiments/sglsim_large/scripts/run_synthesis_smoke.sh
```

The script writes ignored output under:

```text
benchmark/AegisKV/draft/experiments/sglsim_large/results/synthesis_smoke/
```

Each run directory contains `metrics.json`, `request_details.csv`, and
`sglsim_smoke_summary.csv`. These files are smoke artifacts only.

## Policy Candidate Smoke

From the repository root:

```bash
bash benchmark/AegisKV/draft/experiments/sglsim_large/scripts/run_policy_candidate_smoke.sh
```

This writes ignored run output under:

```text
benchmark/AegisKV/draft/experiments/sglsim_large/results/policy_candidate_synthesis/
```

and candidate figure CSVs under:

```text
benchmark/AegisKV/draft/experiments/sglsim_large/results/derived_policy_candidates/
```

Do not copy these CSVs into `figures/data/measured/` as `final`; they are
intermediate evidence for debugging the policy model and figure schemas.

## Failover Replay Smoke

From the repository root:

```bash
bash benchmark/AegisKV/draft/experiments/sglsim_large/scripts/run_failover_replay_smoke.sh
```

This writes ignored run output under:

```text
benchmark/AegisKV/draft/experiments/sglsim_large/results/failover_replay_synthesis/
```

Each run directory contains normal `metrics.json` and `request_details.csv`,
plus `sglsim_failover_replay_summary.csv`. Replay rows are real HTTP requests
through SGLSim serving and are dispatched to a different DP rank from the
failed source rank. In `hard-restart` mode the client calls `/simulate/kill_dp`
before each replay and `/simulate/restart_dp` after it, so the dispatch path
observes a real dead scheduler rank while the replay request is admitted. The
Engine parent looks up the recorded source rank's HostBackup and peer
GPUReplica directory entries, chooses the lower-envelope recovery plan, and
injects the replay restore metadata before dispatch. The scheduler sleeps for
the restore plan before admitting the replay row and materializes restored
prompt-prefix KV as target-rank device-cache metadata. Real CUDA tensor KV copy
is still outside this pure-metadata simulator.

## Failover Replay Matrix

From the repository root:

```bash
python benchmark/AegisKV/draft/experiments/sglsim_large/scripts/run_failover_replay_matrix.py --runs 3
```

The runner creates per-trial config directories with distinct seeds and ports,
launches the existing synthesis_requests serving benchmark, collects per-run
summaries, and then calls `derive_policy_candidates_from_synthesis.py` with all
run directories. The derivation script emits per-figure means and run-level
std/95% CI columns when multiple run directories are supplied. Its candidate CSVs may be copied into
`figures/data/measured/` as preliminary inputs, but they must not be relabeled
as `final` until physical KV movement and final policy sweeps are implemented.

## Large Matrix

`matrix_plan.csv` maps the large SGLSim figures to required simulator
capabilities and measured CSV targets. Current CSVs are preliminary hard
failover replay derivatives; entries remain final-evidence blocked until the
online AegisKV policy path includes physical KV movement and final sweeps.

Large SGLSim figures expected from this path:

- Fig. 3 static GPUReplica budget gap.
- Fig. 8 overall WCT.
- Fig. 10 recovery path breakdown.
- Fig. 10b shared-prefix marginal accounting.
- Fig. 12 ablation.
- Fig. 13 budget sensitivity.
- Fig. 14 DP scale.
- Fig. 14b peer-route selection.
- Fig. 15 simulator calibration anchor; full paired policy calibration still
  requires matched SGLang/SGLSim runs.

Current Fig. 3 uses a replay-calibrated static-budget counterfactual from the
same three hard-failure regimes used by Figs. 2, 8, 9, and 10. It is useful for
internal review, but final evidence still requires online policy sweeps with
physical KV movement.

Current Fig. 12 uses replay-calibrated disabled-signal counterfactuals over the
same regimes. It is useful for ranking likely ablation sensitivity, but final
evidence still requires implemented online policy toggles.

Current Fig. 13 uses replay-calibrated budget-sensitivity rows over the same
three regimes. It replaces the earlier hard-coded heatmap with 264
hard-failure replay rows, but its `failure_timing` axis is still a regime proxy
until a controlled timing sweep exists.

Current Fig. 10b uses original multi-turn request rows from the same matrices
and computes observed safe-frontier marginal bytes per agent. It is an observed
workload-bin figure, not a complete synthetic shared-prefix sweep.

Current Fig. 11b uses direct SGLSim replay ratios for the failure retry-latency
axis over recoverable Host-visible and peer-visible regimes. The normal-path
overhead axis remains preliminary until paired policy measurements exist.
