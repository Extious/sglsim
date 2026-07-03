# SGLSim Direct DP-Scale Sweep Summary

Run id: `sglsim-direct-dp-scale-dp2r5-dp4r3-dp8r3-20260621`

This sweep is the active preliminary input for Fig. 14. It replaces the earlier
DP-scale projection with direct SGLSim serving runs at multiple DP sizes. The
workload remains the standard hard-failover replay workload, so this is still
`evidence_tier=sglsim-synthesis-derived`, not final physical KV-copy evidence.

## Inputs

- DP=2: `sglsim-failover-replay-matrix-r5-20260621`, 5 seeds
- DP=4: `sglsim-dp-scale-dp4-r3-20260621`, 3 seeds
- DP=8: `sglsim-dp-scale-dp8-r3-20260621`, 3 seeds
- Workload: generated shared-prefix, 8 groups, 4 prompts per group, 3 turns
- Failover mode: `hard-restart`
- WCT metric: run duration normalized to the DP=2 mean duration
- Candidate CSV:
  `results/failover_replay_matrix/direct_dp_scale_20260621/fig_dp_scale.csv`

## Results

- DP=2: 5/5 seeds completed, 600/600 rows, 120 hard-failure replay rows,
  normalized WCT `1.000`
- DP=4: 3/3 seeds completed, 360/360 rows, 72 hard-failure replay rows,
  normalized WCT `0.985`
- DP=8: 3/3 seeds completed, 360/360 rows, 72 hard-failure replay rows,
  normalized WCT `0.983`

## Interpretation

The direct sweep validates that the current SGLSim hard-failover replay path
can run at DP=2, DP=4, and DP=8 with cross-rank killed-scheduler retry rows and
without failed requests. Under this low-concurrency workload, duration is
nearly flat across DP sizes; this is useful as a direct serving-path scale
smoke, but it does not yet demonstrate large-workload scaling.

Final Fig. 14 still requires high-concurrency online AegisKV policy sweeps with
physical KV movement and repeated workload-concurrency settings.
