# Figure Result CSV Contracts

Measured CSVs must be placed under `benchmark/AegisKV/presentation/figures/data/measured/`.
The filename must match the figure stem in `figures/manifest.json`.

| Figure stem | Required columns |
|---|---|
| `fig_motivation_equal_size_loss` | `segment`, `delta_wct_norm` |
| `fig_critical_backup_priority` | `backup_budget_fraction`, `placement_policy`, `normalized_wct_after_failure` |
| `fig_critical_agent_profile` | `stage_slot`, `stage_id`, `stage_width`, `stage_local_agent_index`, `critical_agent`, `decode_dominant`, `mean_output_tokens`, `mean_latency_ms`, `stage_critical_path_share` |
| `fig_restore_recompute_boundary` | `tokens`, `recompute_ms`, `host_restore_ms`, `gpu_replica_ms`, `aegis_lower_envelope_ms` |
| `fig_static_gpu_replica_budget_gap` | `gpu_replica_budget_fraction`, `workload`, `normalized_wct` |
| `fig_alpha_workload_failure_policy` | `failure_rate_per_window`, `critical_pressure`, `alpha_backup_fraction` |
| `fig_gpu_replica_alpha_tradeoff` | `concurrency`, `alpha`, `normalized_wct`, `normalized_failure_wct`, `alpha_star` |
| `fig_online_alpha_adaptation` | `request_class`, `failure_rate_per_window`, `critical_pressure`, `policy_alpha`, `effective_gpu_replica_alpha`, `online_observed_alpha`, `observed_request_rows` |
| `fig_staged_alpha_risk_tradeoff` | `failure_rate_per_window`, `alpha_backup_fraction`, `normalized_expected_wct`, `normal_backup_overhead_s`, `expected_failure_recovery_s`, `protected_critical_token_fraction`, `optimal_alpha_for_rate` |
| `fig_overall_wct` | `policy`, `normalized_wct`, `failure_exposed_restore_norm` |
| `fig_recovery_latency` | `policy`, `retry_latency_norm`, `critical_agent_latency_norm` |
| `fig_recovery_path_breakdown` | `policy`, `peer_gpu`, `host_h2d`, `ordinary_cache`, `recompute` |
| `fig_shared_prefix_marginal` | `shared_prefix_pct`, `dup_no_dedup`, `dup_marginal`, `coverage_marginal` |
| `fig_normal_path_overhead` | `policy`, `p99_overhead_pct`, `write_amplification` |
| `fig_hostbackup_visibility` | `policy`, `p99_overhead_relative_pct`, `retry_latency_norm` |
| `fig_ablation` | `variant`, `normalized_wct` |
| `fig_budget_sensitivity` | `failure_timing`, `gpu_replica_budget_fraction`, `normalized_wct` |
| `fig_dp_scale` | `dp_size`, `workflow_count`, `normalized_wct` |
| `fig_peer_rank_selection` | `policy`, `normalized_wct`, `peer_p99_overhead_pct` |
| `fig_simulator_calibration` | `policy`, `sglang_norm`, `sglsim_norm` |
| `fig_physical_recovery_copy_calibration` | `path`, `tokens`, `p50_ms`, `p50_ms_ci95` |
| `fig_recovery_model_calibration_error` | `path`, `tokens`, `measured_copy_ms`, `modeled_copy_ms`, `abs_error_ms`, `rel_error_pct` |
| `fig_sglang_sglsim_accuracy` | `metric`, `sglang_value`, `sglsim_value`, `ratio`, `rel_error_pct`, `within_10pct` |

Additional provenance columns are recommended for every measured CSV:

```text
evidence_tier, source, run_id, git_sha, remote_host, slurm_job_id, model_path,
workload_id, dp_size, tp_size, failure_rank, failure_timing, repeat
```

`evidence_tier` is mandatory in practice even though the plotting code permits
older files:

- `sglsim-preliminary`: deterministic counterfactual or smoke-derived data for
  internal paper development.
- `final`: final SGLang/SGLSim experiment data that can support HPCA
  quantitative claims.

The validator permits additional columns, but rejects missing required columns
or empty measured files. `--strict-final` requires every result figure to have
valid rows with `evidence_tier=final`.
