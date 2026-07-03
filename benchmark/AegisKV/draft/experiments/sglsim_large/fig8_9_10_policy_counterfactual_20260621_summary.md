# Fig. 8/9/10 Multi-Policy Replay Counterfactual

- run_id: `sglsim-fig8-10-policy-counterfactual-r11-20260621`
- source: `sglsim_multiregime_replay_policy_counterfactual`

## Fig. 8

| policy | primary metric | secondary |
|---|---:|---:|
| no backup | 1.0000 | 0.0000 |
| uniform Host | 0.9925 | 0.0554 |
| uniform GPU | 0.9842 | 0.0158 |
| AegisKV | 0.9716 | 0.0369 |

## Fig. 9

| policy | primary metric | secondary |
|---|---:|---:|
| no backup | 1.0000 | 1.0000 |
| uniform Host | 1.1100 | 1.2574 |
| uniform GPU | 0.8300 | 0.8526 |
| AegisKV | 0.7571 | 0.8025 |

## Fig. 10

| policy | primary metric | secondary |
|---|---:|---:|
| no backup | 1.0000 | 0.0000 |
| uniform Host | 0.0000 | 1.0000 |
| uniform GPU | 0.8602 | 0.1398 |
| AegisKV | 0.5692 | 0.4308 |

These are replay-calibrated counterfactual rows, not final physical KV-copy policy sweeps.
