# SGLSim Failover Replay Matrix R3 Summary

Run id: `sglsim-failover-replay-matrix-r3-20260621`

This run uses `scripts/run_failover_replay_matrix.py` to repeat the
hard-failover replay benchmark across three seeds. It is stronger than the
single failover replay smoke because the generated figure CSVs aggregate
multiple run directories, but it is still preliminary evidence:
`evidence_tier=sglsim-synthesis-derived`.

This is now a historical intermediate matrix. The active preliminary figure
package has been superseded by
`sglsim-failover-replay-matrix-r5-20260621`.

## Inputs

- Base config: `config_failover_replay/`
- Seeds: `8200`, `8201`, `8202`
- DP size: `2`
- Model path: `mock-model`
- Workload: generated shared-prefix, 8 groups, 4 prompts per group, 3 turns
- Failover mode: `hard-restart`
- Candidate CSV directory:
  `results/failover_replay_matrix/sglsim-failover-replay-matrix-r3-20260621/derived_policy_candidates_sglang6_ci/candidate_measured/`
  This was the active figure-data package before the five-seed r5 matrix. It
  kept the three-run SGLSim failover replay aggregation and injected the
  seven-job SGLang DP=2 paired Fig. 11 overhead aggregate for
  `sglang-small-paired` rows.

## Results

- Completed request rows: `360`
- Failed request rows: `0`
- Candidate retry rows: `144`
- Real HTTP failover replay rows: `72`
- Cross-rank replay rows: `72`
- Hard-failure injected replay rows: `72`
- HostBackup lookup-hit replay rows: `72`
- Peer GPUReplica lookup-hit replay rows: `72`
- HostBackup lookup tokens: `123579`
- Peer GPUReplica lookup tokens: `15685`
- Restored prompt-prefix tokens materialized as target-rank cache: `121389`
- Replay recompute tokens: `0`
- Mean replay restore sleep: `13.74 ms`

## Interpretation

The run validated the SGLSim serving path for repeated killed-scheduler
cross-rank retry-shaped requests with runtime HostBackup and peer GPUReplica
directory lookup. The replay target rank materializes restored prompt-prefix KV
as metadata before replay admission, and the scheduler sleeps for the restore
plan, so restore cost affects observed wall-clock latency.

It does not yet execute real CUDA tensor KV copy. Therefore the generated CSVs
may refresh preliminary figures, but they must not be relabeled as final HPCA
evidence.
