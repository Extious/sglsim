# SGLSim Failover Replay Matrix R5 Summary

Run id: `sglsim-failover-replay-matrix-r5-20260621`

This run repeats the hard-failover replay benchmark across five seeds. It is
the active preliminary SGLSim input for the current figure package. The run is
stronger than the earlier three-seed matrix because the selected figure CSVs
now aggregate five run-level candidate derivations as mean/std/95% CI, but it
is still preliminary evidence: `evidence_tier=sglsim-synthesis-derived`.

## Inputs

- Base config: `config_failover_replay/`
- Seeds: `9300`, `9301`, `9302`, `9303`, `9304`
- DP size: `2`
- Model path: `mock-model`
- Workload: generated shared-prefix, 8 groups, 4 prompts per group, 3 turns
- Failover mode: `hard-restart`
- Candidate CSV directory:
  `results/failover_replay_matrix/sglsim-failover-replay-matrix-r5-20260621/derived_policy_candidates/candidate_measured/`
- Figure-package note: Fig. 14 (`fig_dp_scale.csv`) is now replaced by the
  direct DP-scale sweep in `sglsim_direct_dp_scale_20260621_summary.md`. The
  older r5-derived scale-model projection is kept only as derivation history.

## Results

- Completed request rows: `600`
- Failed request rows: `0`
- Candidate retry rows: `240`
- Real HTTP failover replay rows: `120`
- Cross-rank replay rows: `120`
- Hard-failure injected replay rows: `120`
- HostBackup lookup-hit replay rows: `120`
- Peer GPUReplica lookup-hit replay rows: `120`
- HostBackup lookup tokens: `201787`
- Peer GPUReplica lookup tokens: `25592`
- Restored prompt-prefix tokens materialized as target-rank cache: `198322`
- Replay recompute tokens: `0`
- Mean replay restore sleep: `13.62 ms`

## Interpretation

The run validates the current SGLSim serving path for repeated killed-scheduler
cross-rank retry-shaped requests with runtime HostBackup and peer GPUReplica
directory lookup. The active figure package now uses these five repeated runs
for the SGLSim rows and appends the seven-job SGLang DP=2 paired aggregate for
Fig. 11.

It still does not execute real CUDA tensor KV copy or final online AegisKV
policy sweeps. Therefore the generated CSVs may refresh preliminary figures,
but they must not be relabeled as final HPCA evidence.
