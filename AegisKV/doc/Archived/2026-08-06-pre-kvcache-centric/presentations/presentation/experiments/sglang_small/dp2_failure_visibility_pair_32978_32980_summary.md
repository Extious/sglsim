# DP=2 Failure Visibility Pair 32978/32980

Purpose:

- compare two real SGLang DP=2 failure-control smokes with the same high-level
  request shape;
- keep Host-invisible retry continuity evidence separate from Host-visible
  remote-backup frontier evidence;
- provide a machine-readable support artifact for Fig. 11b and Fig. 15 without
  promoting it to final HPCA evidence.

Pair:

| Role | Slurm job | Storage backend | Requests | Failover responses | Pre-failover output tokens | Pre-failover backed-up tokens | p99 latency |
|---|---:|---|---:|---:|---:|---:|---:|
| Host-invisible control path | 32978 | none | 4 | 4 | 760 | 0 | 16.893 s |
| Host-visible frontier smoke | 32980 | remote_backup | 4 | 4 | 648 | 5,132 | 18.715 s |

Derived checks:

- Host frontier delta: 5,132 backed-up tokens.
- Remote-backup frontier density: 1,283 backed-up tokens/request.
- Remote-backup frontier density per pre-failover output token: 7.92x.
- Remote-backup p99 latency normalized to the no-storage control path: 1.108x.

Interpretation:

The pair proves that the real SGLang failure-control path can both survive
DP-rank failure and report a nonzero Host-visible frontier when a storage
backend is attached. It does not yet prove that Host-visible restore improves
retry latency, because this smoke uses only four requests, disables overlap
scheduling for control-path stability, and still lacks a matched final
AegisKV restore-policy comparison.

Artifacts:

- `results/failure_visibility_pair_32978_32980/failure_visibility_pair_summary.json`
- `results/failure_visibility_pair_32978_32980/failure_visibility_pair_summary.csv`
- `results/failure_visibility_pair_32978_32980/failure_visibility_pair_rows.csv`
- `results/failure_visibility_pair_32978_32980/candidate_fig_hostbackup_visibility_sglang_pair.csv`
