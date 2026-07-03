# AegisKV Experiments

This directory is the evidence bridge between the draft and the figure package.
It separates three artifact classes:

- `sglang_small/`: real SGLang experiments on `hkbugpumanode` through Slurm.
  The current strict-final recovery matrix plan is generated under
  `sglang_small/plans/strict_final_recovery_matrix/`.
- `sglsim_large/`: large online SGLSim sweeps after AegisKV hooks are restored or implemented.
- `schema/`: CSV contracts used by `../figures/scripts/validate_figure_evidence.py`.
- `final_evidence_matrix.csv`: per-figure map from preliminary artifacts to
  the SGLang/SGLSim measurements required before a result can be marked
  `evidence_tier=final`.
- `strict_final_experiment_gap.md`: prioritized Chinese checklist for the
  remaining physical-movement / real-recovery evidence needed after the current
  HPCA hybrid gate passes.
- `sglsim_large/results/strict_final_readiness/p0_strict_final_readiness.md`:
  generated audit for the current P0 online suites; it reports how many restore
  rows are still metadata-only versus `physical_tensor_copy`.
- `goal_completion_audit.md`: generated audit for the active AegisKV objective
  across workflow shape, motivation figures, MDP/online alpha evidence,
  SGLang/SGLSim accuracy, Beamer assets, and the P0 strict-final data plane.

Measured data that should replace expected-trend templates must be exported as:

```text
benchmark/AegisKV/draft/figures/data/measured/<figure_stem>.csv
```

The figure stems are listed in `../figures/manifest.json`. A result figure is
not claim-ready until its measured CSV passes the validator.

`final_evidence_matrix.csv` is the planning source for the final evidence
campaign. It intentionally lists the quantitative result figures, including
figures whose final source is primarily SGLang small-scale measurement
(for example Fig. 11), figures whose final source is primarily online
SGLSim scale sweeps, and trace-derived motivation figures that still need
stronger final calibration before submission.

Current non-final evidence levels:

- `sglsim-preliminary`: deterministic counterfactual CSVs kept as historical
  scaffolding and template support.
- `sglsim-synthesis-derived`: candidate CSVs derived from real SGLSim
  `synthesis_requests` serving runs plus a post-run AegisKV policy model.
  The current preliminary figure package uses the five-seed hard-failover
  replay matrix variant for large-scale SGLSim result figures. These CSVs are
  useful for debugging figure schemas and policy equations. They may be
  mirrored into `../figures/data/measured/` for preliminary plotting only, but
  must keep a non-final evidence tier.
- `sglang-small-paired`: small real SGLang paired measurements from
  `hkbugpumanode`. The current Fig. 11 package uses seven DP=2 paired Slurm
  jobs as calibration/provenance, but those runs are not enough by themselves
  for final HPCA claims.
- `final`: the only evidence tier accepted by `--strict-final`.

## Validation

```bash
python benchmark/AegisKV/draft/figures/scripts/generate_all_figures.py
python benchmark/AegisKV/draft/figures/scripts/validate_figure_evidence.py
python benchmark/AegisKV/draft/figures/scripts/validate_figure_evidence.py --write-hpca-hybrid
```

The HPCA hybrid report writes:

- `../figures/hpca_hybrid_evidence_status.json`
- `hpca_hybrid_evidence_status.md`

This gate is weaker than `--strict-final`. It accepts selected SGLSim result
figures only when the small physical recovery microbench is present and the
metadata-only SGLSim data plane is explicitly disclosed.

Before treating the draft as submission-ready:

```bash
python benchmark/AegisKV/draft/figures/scripts/validate_figure_evidence.py --strict-final
python benchmark/AegisKV/draft/figures/scripts/validate_figure_evidence.py --strict-hpca-hybrid
```

The strict check is expected to fail until all result figures have measured
SGLang/SGLSim CSVs. The hybrid strict check is expected to fail until all
remaining replay counterfactuals have been replaced by implemented online
policy sweeps or paired SGLang/SGLSim measurements.

To inspect the current per-figure blockers:

```bash
python benchmark/AegisKV/draft/scripts/report_final_gate_blockers.py
```

This writes `final_gate_blockers.md` and `final_gate_blockers.json` in this
directory. The report is generated from the figure evidence status, the final
evidence matrix, and the SGLSim matrix plan.

To inspect the active objective as a requirement checklist:

```bash
python benchmark/AegisKV/draft/scripts/audit_goal_completion.py
```

This writes `goal_completion_audit.md` and `goal_completion_audit.json`. A
current `blocked` status is expected until the P0 policy/recovery suites report
live `physical_tensor_copy` recovery rows or an equivalent validated real-system
recovery path.

To generate the real-system recovery-path matrix needed for that blocker:

```bash
python benchmark/AegisKV/draft/experiments/sglang_small/scripts/plan_strict_final_recovery_matrix.py
```

This writes submit commands, pair-sync commands, and physical-restore parser
commands under `sglang_small/plans/strict_final_recovery_matrix/`. The plan is
not a result artifact; it preserves the boundary that SGLSim restore rows remain
metadata-only until real SGLang recovery evidence is collected and explicitly
connected to the final claims.

After SGLang pair results are synced, audit the matrix with:

```bash
python benchmark/AegisKV/draft/experiments/sglang_small/scripts/audit_strict_final_recovery_matrix.py
```

This writes `sglang_small/results/strict_final_recovery_matrix_audit/`. The
current report can be `partial`: best-effort recovery-benefit pairs and
wait-complete materialization pairs are tracked independently, and a `pass`
there still does not by itself turn SGLSim P0 metadata rows into
`physical_tensor_copy` rows. When the report is partial, use
`strict_final_recovery_missing_jobs.csv` in that directory for the next Slurm
submit/sync commands. The audit reads both the aggregate
`physical_restore_evidence/physical_restore_runs.csv` and per-pair
`physical_restore_evidence/*/physical_restore_runs.csv` files, so synced
strict-final rerun results do not need to be manually merged into the primary
CSV. To materialize executable helper
scripts from that CSV:

```bash
python benchmark/AegisKV/draft/experiments/sglang_small/scripts/prepare_strict_final_missing_jobs.py
```

After `sbatch` returns job ids, bind them and generate the runnable sync script:

```bash
python benchmark/AegisKV/draft/experiments/sglang_small/scripts/bind_strict_final_job_ids.py --submit-log <submit_log>
```
