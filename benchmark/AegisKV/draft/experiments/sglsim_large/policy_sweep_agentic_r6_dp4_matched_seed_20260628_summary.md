# SGLSim Agentic Policy Sweep r6

Run id: `sglsim-policy-sweep-agentic-r6-dp4-matched-seed-20260628`

Purpose: replace Fig. 8's replay counterfactual with an implemented online
SGLSim policy sweep using explicit `gsp_aegis_policy`.

Configuration:

- DP size: 4
- Workload: 32 generated shared-prefix agents, 3 turns
- Shape: `system_prompt_len=512`, `question_len=256`, `output_len=64`
- Matched seeds: `8100`, `8101`, `8102`
- Failure mode: hard-restart replay, retry rank offset 1
- Recovery timing: `SGLSIM_AEGIS_RECOVERY_MODEL=physical_copy` with the
  recovery-shaped physical movement aggregate from jobs `34357/34358/34359`

Results:

| Policy | Completed | Failed | Replay rows | Restore actions | Mean E2E ms | Normalized WCT |
|---|---:|---:|---:|---|---:|---:|
| no backup | 360 | 0 | 72 | 72 recompute | 107.763 | 1.000 |
| uniform Host | 360 | 0 | 72 | 72 Host restore | 105.850 | 0.982 |
| uniform GPUReplica | 360 | 0 | 72 | 72 peer-GPU restore | 105.844 | 0.982 |
| AegisKV | 360 | 0 | 72 | 48 Host restore, 24 peer-GPU restore | 105.218 | 0.976 |

Evidence status:

- Fig. 8 measured CSV:
  `benchmark/AegisKV/draft/figures/data/measured/fig_overall_wct.csv`
- Evidence tier: `sglsim-agentic-online-policy-sweep`
- Hybrid gate status: `hybrid-online-policy-sweep`
- Caveat: SGLSim remains metadata-only. Restore timing is calibrated from the
  physical recovery-action microbench, but the online rows do not execute live
  CUDA tensor movement.
