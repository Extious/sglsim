# AegisKV Final Gate Blockers

This report is generated from `figures/evidence_status.json`, `experiments/final_evidence_matrix.csv`, and `experiments/sglsim_large/matrix_plan.csv`.

It tracks why `validate_figure_evidence.py --strict-final` is still blocked. Mechanism figures are omitted because they do not require measured CSVs.

## Summary

- Figures: 25
- Claim-ready figures: 2
- Quantitative result figures: 24
- Final measured result figures: 1
- Preliminary valid result figures: 23
- Blocked result figures: 23
- Resolved result figures: 1
- HPCA hybrid-ready figures: 25
- HPCA hybrid-ready result figures: 24
- HPCA hybrid-blocked result figures: 0

## Evidence Boundaries

- `--strict-final` accepts only `evidence_tier=final` result CSVs.
- The HPCA hybrid report accepts selected calibrated SGLSim figures only when small physical movement measurements are present and the metadata-only data plane is disclosed.
- SGLSim remains a pure metadata online simulator; live CUDA page movement must stay in SGLang/torch calibration harnesses or a separate real-system path.

## Blocker Classes

- final measured CSV replacement: 6
- paired SGLang/SGLSim policy measurements: 1
- physical KV movement / CUDA copy validation: 13
- unclassified final-evidence gap: 3

## HPCA Hybrid Blocker Classes


## Per-Figure Gate

| Figure | Stem | Current tier | Rows | Hybrid | Primary final source | Status | Final gate |
|---|---|---|---:|---|---|---|---|
| Fig. 1 | `fig_motivation_equal_size_loss` | `sglsim-trace-derived-stagebarrier` | 5 | hybrid-trace-derived-motivation | sglsim_large | final measured CSV replacement | hybrid-ready motivation evidence; strict-final still needs final measured workflow-loss calibration |
| Fig. 1b | `fig_critical_backup_priority` | `sglsim-trace-derived` | 44 | hybrid-trace-derived-motivation | sglsim_large | final measured CSV replacement | hybrid-ready motivation evidence; strict-final still needs real-system failure calibration |
| Fig. 1c | `fig_critical_agent_profile` | `sglsim-trace-derived-stagebarrier` | 9 | hybrid-trace-derived-motivation | sglsim_large | final measured CSV replacement | hybrid-ready motivation evidence; strict-final still needs real-system calibration of the same agentic workflow |
| Fig. 2 | `fig_restore_recompute_boundary` | `sglsim-runtime-replay-calibrated` | 8 | hybrid-physical-calibrated | sglang_small+sglsim_large | physical KV movement / CUDA copy validation | paired SGLang microbench plus calibrated SGLSim sweep; final still needs end-to-end peer GPU/P2P physical-copy recovery validation |
| Fig. 3 | `fig_static_gpu_replica_budget_gap` | `sglsim-agentic-online-budget-sweep` | 16 | hybrid-online-policy-sweep | sglsim_large | physical KV movement / CUDA copy validation | hybrid-ready via implemented online budget sweep; strict-final still needs physical KV movement rather than metadata materialization |
| Fig. 3b | `fig_alpha_workload_failure_policy` | `sglsim-model-policy-table` | 15 | hybrid-model-policy | sglsim_model | physical KV movement / CUDA copy validation | hybrid-ready model-policy evidence; Fig. 3c and runtime replay verify online use; strict-final still needs physical KV movement or final measured adaptive-alpha policy evidence |
| Fig. 3c | `fig_online_alpha_adaptation` | `sglsim-agentic-online-alpha-sweep` | 8 | hybrid-online-alpha-sweep | sglsim_large | physical KV movement / CUDA copy validation | hybrid-ready online adaptive-alpha serving sweep; strict-final still needs physical KV movement evidence for the same policy path |
| Fig. 3d | `fig_staged_alpha_risk_tradeoff` | `sglsim-trace-derived` | 55 | hybrid-trace-derived-motivation | sglsim_large | physical KV movement / CUDA copy validation | hybrid-ready staged alpha-risk motivation evidence; strict-final still needs repeated online adaptive-alpha serving sweep with physical KV movement |
| Fig. 8 | `fig_overall_wct` | `sglsim-agentic-online-policy-sweep` | 4 | hybrid-online-policy-sweep | sglsim_large | physical KV movement / CUDA copy validation | final policy matrix with confidence intervals and no failed-request artifacts |
| Fig. 9 | `fig_recovery_latency` | `sglsim-agentic-online-recovery-latency-sweep` | 4 | hybrid-paired-anchor | sglsim_large | physical KV movement / CUDA copy validation | hybrid-ready via online SGLSim recovery-latency sweep plus SGLang paired anchors; strict-final still needs physical KV movement rather than metadata materialization |
| Fig. 10 | `fig_recovery_path_breakdown` | `sglsim-agentic-online-preliminary,sglsim-runtime-replay-calibrated` | 8 | hybrid-physical-calibrated | sglsim_large | physical KV movement / CUDA copy validation | path shares must sum to one per policy and be backed by simulator event counters; final rows require end-to-end physical CUDA KV-copy recovery not metadata-only restoration |
| Fig. 10b | `fig_shared_prefix_marginal` | `final` | 6 | strict-final-ready | sglsim_large | resolved | satisfied by final content-frontier accounting sweep over low/mid/high shared-prefix scenarios |
| Fig. 11 | `fig_normal_path_overhead` | `sglang-small-paired,sglsim-agentic-online-preliminary,sglsim-synthesis-derived` | 6 | hybrid-paired-anchor | sglang_small+sglsim_large | physical KV movement / CUDA copy validation | final CSV must include repeated matched SGLang evidence and SGLSim scale runs with physical KV movement; current online sweep strengthens the no-failure overhead anchor but is still not full AegisKV policy evidence |
| Fig. 11b | `fig_hostbackup_visibility` | `mixed-preliminary,sglang-small-paired-smoke` | 8 | hybrid-paired-anchor | sglang_small+sglsim_large | paired SGLang/SGLSim policy measurements | final rows must prove same normal D2H cost but different failure visibility with paired policy measurements and repeated policy measurements |
| Fig. 12 | `fig_ablation` | `sglsim-agentic-online-ablation-sweep` | 7 | hybrid-online-policy-sweep | sglsim_large | physical KV movement / CUDA copy validation | hybrid-ready via implemented online toggles; strict-final still needs physical KV movement rather than metadata materialization |
| Fig. 13 | `fig_budget_sensitivity` | `sglsim-agentic-online-budget-timing-sweep` | 15 | hybrid-online-policy-sweep | sglsim_large | physical KV movement / CUDA copy validation | hybrid-ready via implemented online budget/timing sweep; strict-final still needs physical KV movement rather than metadata materialization |
| Fig. 14 | `fig_dp_scale` | `sglsim-agentic-online-preliminary,sglsim-runtime-replay-derived` | 7 | hybrid-physical-calibrated | sglsim_large | physical KV movement / CUDA copy validation | final data must add high-concurrency repeated online AegisKV sweeps with physical KV movement |
| Fig. 14b | `fig_peer_rank_selection` | `sglsim-agentic-online-peer-route-sweep` | 3 | hybrid-online-policy-sweep | sglsim_large | physical KV movement / CUDA copy validation | final rows must expose WCT recovery overhead peer-hit rate and load pressure from implemented policy sweeps |
| Fig. 15 | `fig_simulator_calibration` | `mixed-preliminary` | 4 | hybrid-paired-anchor | sglang_small+sglsim_large | final measured CSV replacement | final claim is ranking preservation; current anchor compares normal-path p99 overhead plus failover frontier continuity |
| Fig. 15b | `fig_physical_recovery_copy_calibration` | `sglang-small-physical-recovery-movement-smoke` | 15 | hybrid-physical-calibrated | sglang_small | unclassified final-evidence gap | hybrid calibration anchor; not end-to-end policy recovery evidence |
| Fig. 15c | `fig_recovery_model_calibration_error` | `sglsim-model-physical-copy-calibration` | 10 | hybrid-physical-calibrated | sglsim_model+sglang_small | unclassified final-evidence gap | hybrid calibration-wiring evidence; not an independent end-to-end accuracy claim because the model reads the physical CSV |
| Fig. 15d | `fig_sglang_sglsim_accuracy` | `sglang-sglsim-synthesis-parity` | 8 | hybrid-simulator-accuracy | sglang_small+sglsim_large | unclassified final-evidence gap | hybrid simulator-accuracy evidence for base serving timing; not end-to-end AegisKV failure-policy evidence |
| Fig. 15e | `fig_p0_physical_bridge_coverage` | `sglsim-physical-bridge-coverage` | 2 | hybrid-physical-calibrated | sglsim_large+sglang_small | final measured CSV replacement | hybrid calibration-coverage evidence; strict-final still requires live physical_tensor_copy rows or equivalent end-to-end recovery path |
| Fig. 15f | `fig_sglang_host_restore_materialization` | `sglang-small-physical-restore-materialization` | 3 | hybrid-sglang-host-restore | sglang_small | final measured CSV replacement | hybrid real-system Host restore materialization evidence; strict-final still requires P0 live physical_tensor_copy rows and peer GPUReplica/P2P policy evidence |

## Next Practical Milestone

The HPCA hybrid report has no remaining blocked figures. The next paper-strengthening step is strict-final evidence: replace preliminary SGLSim metadata-materialization rows with final end-to-end measurements that use physical KV movement or explicitly validated real-system recovery paths.
