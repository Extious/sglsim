# SGLSim Agentic Peer-Route Sweep r1

Run id: `sglsim-peer-route-sweep-agentic-r1-dp4-matched-seed-20260628`

Purpose: replace Fig. 14b's matched replay derivation with implemented online
SGLSim retry-route policies.

Configuration:

- DP size: 4
- Workload: 32 generated shared-prefix agents, 3 turns
- Shape: `system_prompt_len=64`, `question_len=32`, `output_len=64`
- Matched seeds: `8500`, `8501`, `8502`
- Failure mode: hard-restart replay
- Route policies: `non_peer`, `random_healthy`, `peer`
- Recovery timing: `SGLSIM_AEGIS_RECOVERY_MODEL=physical_copy` with the
  recovery-shaped physical movement aggregate from jobs `34357/34358/34359`

Results:

| Policy | Completed | Failed | Replay rows | Peer hits | Restore actions | Mean E2E ms | Normalized WCT | p99 overhead |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| non-peer route miss | 360 | 0 | 72 | 0 | 72 recompute | 94.529 | 1.000 | 100.0% |
| random healthy route | 360 | 0 | 72 | 21 | 21 GPUReplica, 51 recompute | 94.759 | 1.002 | 99.2% |
| AegisKV peer route | 360 | 0 | 72 | 72 | 72 GPUReplica | 93.366 | 0.988 | 74.3% |

Evidence status:

- Fig. 14b measured CSV:
  `benchmark/AegisKV/draft/figures/data/measured/fig_peer_rank_selection.csv`
- Evidence tier: `sglsim-agentic-online-peer-route-sweep`
- Hybrid gate status: `hybrid-online-policy-sweep`
- Caveat: SGLSim remains metadata-only. Restore timing is calibrated from the
  physical recovery-action microbench, but the online rows do not execute live
  CUDA tensor movement.
