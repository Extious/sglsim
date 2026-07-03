# SGLSim Physical-Copy-Calibrated Restore Smoke

This artifact validates the SGLSim runtime switch that uses the repeated
SGLang physical CUDA copy aggregate for recovery timing. It is not end-to-end
physical CUDA KV movement evidence: SGLSim still materializes restored prefixes
as metadata, but `restore_overhead_s` is computed from the measured H2D/P2P
copy curves when the physical-copy model is enabled.

## Run

- Run id: `sglsim-physical-copy-model-smoke-20260628`
- Run directory:
  `results/agentic_workflows/sglsim-physical-copy-model-smoke-20260628/runs/trial_000/20260628_031505`
- Scenario: `physical_copy_restore_smoke`
- Workload: DP=2, 2 workflows, 2 turns, hard-restart failover replay
- Environment:
  - `SGLSIM_AEGIS_RECOVERY_MODEL=physical_copy`
  - `SGLSIM_AEGIS_PHYSICAL_COPY_CSV=/home/comp/zhanzhao/sglsim/benchmark/AegisKV/draft/experiments/sglang_small/results/physical_kv_copy_aggregate/physical_kv_copy_aggregate.csv`

## Result

The run completed all requests and replay rows:

| Metric | Value |
|---|---:|
| completed rows | 8 |
| failed rows | 0 |
| failover replay rows | 4 |
| hard-failure replay rows | 4 |
| cross-rank replay rows | 4 |
| Host lookup hits | 4 |
| peer lookup hits | 4 |
| restored tokens | 106 |
| restored prefill tokens | 106 |
| recompute tokens | 738 |
| mean restore sleep | 4.466 ms |

All four replay rows choose `gpu_replica`. The physical-copy-calibrated
per-replay restore overheads are in the 4.07-4.78 ms range because the restored
peer GPUReplica frontier is small (22-31 tokens) and the remaining suffix is
still modeled as recompute.

## Interpretation

This closes a wiring gap: SGLSim can now run with physical-copy-calibrated
Host/GPUReplica restore timing while preserving the same online
request/replay/failure path. It does not close the final HPCA gate because the
runtime still performs metadata materialization rather than CUDA tensor KV
movement. The next required step is to connect equivalent physical movement to
real SGLang/SGLang-small recovery or to an online SGLSim path that executes
CUDA copy operations for the restored pages.
