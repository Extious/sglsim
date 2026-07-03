# SGLSim Shared-Prefix Marginal Accounting Summary

Run id: `sglsim-fig10b-shared-prefix-marginal-r11-20260621`

This artifact is the active preliminary input for Fig. 10b. It replaces the
older synthetic shared-prefix sweep with observed multi-turn request bins from
the current SGLSim hard-failure matrices.

## Inputs

- Source matrix 1: `sglsim-failover-replay-matrix-r5-20260621`
- Source matrix 2: `sglsim-fig10-peer-replay-r3-20260621`
- Source matrix 3: `sglsim-fig10-recompute-replay-r3-20260621`
- Run directories: `11`
- Original request rows used: `1056`
- Multi-turn agent groups observed: `352`
- Agent groups retained after min-count bin filtering: `348`
- Derivation script: `scripts/derive_shared_prefix_marginal_from_replay.py`
- Output CSV: `figures/data/measured/fig_shared_prefix_marginal.csv`

## Method

For each run and agent, original request rows are sorted by `aegis_worker_seq`.
Retry rows are excluded. The no-dedup baseline sums every turn's
`aegis_safe_frontier_tokens`. Marginal accounting counts the first frontier and
then only positive frontier growth on later turns. Values are normalized by the
latest safe frontier in that agent group.

## Results

Observed shared-prefix bins are `42.5%`, `47.5%`, `52.5%`, and `57.5%`.

- No-dedup protected bytes rise from `2.07x` to `2.39x` the latest frontier.
- Marginal protected bytes stay near `1.0x` to `1.20x`.
- Reconstruction coverage is `1.0` in retained bins because the marginal
  frontier increments reconstruct the latest safe frontier.

## Interpretation

Fig. 10b now supports a direct current-workload claim: multi-turn workflow
frontiers contain substantial duplicated safe-prefix bytes, and marginal
accounting avoids most duplicate protection without losing the final frontier.
This is not yet the final broad sweep: the observed workload covers about
`42-58%` shared prefix after bin filtering, so a final paper run should add
controlled 0-95% shared-prefix sweeps.
