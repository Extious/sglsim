# Latest AegisKV independent-tier simulation

> **Historical mechanism prototype (status clarified 2026-08-27).** A later
> self-audit found that this first study did not yet exercise progressive
> runtime chunk sealing, independent transfer timing/failure, final request
> cleanup, the complete Engine failure/rebind path, or Algorithm 1 in the
> end-to-end run. Its synthetic percentages remain reproducible mechanism
> diagnostics, but they are not evidence of a complete current-design runtime
> or a current paper result. Start from
> [`../../PROJECT_STATUS.md`](../../PROJECT_STATUS.md) and
> [`../README.md`](../README.md) when choosing an experiment lineage.

This directory is the reproducible SGLSim mechanism study for
`source_owned_independent_hbm_dram_v2`. It exercises
`AegisIndependentTierController` directly and compares the same seeded request,
copy-completion, target-failure, source-failure, Slack, capacity, and AIMD
events under five policies:

- `recompute_only`;
- capacity-matched `static_independent_fcfs` without SlackLandlord;
- HBM-only and DRAM-only ablations;
- the latest independent HBM + DRAM AegisKV policy.

The workload has 32 matched seeds, 96 active requests per seed, one source
failure per seed, a shared HBM-pressure event, independent HBM/DRAM copy
outcomes, and a 25% probability that the bound HBM target fails before the
source. HBM and DRAM are overlapping recovery alternatives: the audit rejects
any result that adds their token counts as disjoint ranges.

Run from the repository root:

```bash
uv run python AegisKV/experiments/latest_design/run_simulation.py
uv run python AegisKV/experiments/latest_design/plot_results.py
uv run python AegisKV/experiments/latest_design/run_engine_smoke.py
```

The checked-in result contains 3,072 matched request failures per policy. The
latest design reduces mean Slack-visible recovery harm by **18.6%** versus
recompute-only (seed-bootstrap 95% CI: **18.2%--19.0%**) and by **5.6%** versus
the capacity-matched static independent-tier policy (95% CI:
**5.1%--6.3%**). Mean recovery time changes from 70.1 ms (recompute) and 61.7 ms
(static) to 61.5 ms (latest); the larger harm improvement over static reflects
Slack-aware placement concentrating protection on failures that would be
visible to workflow completion.

Outputs:

- `results/raw_failures.csv`: matched failure-level rows;
- `results/episode_stats.csv`: normal-path modeled copy work and pool state;
- `results/summary.json` / `summary.csv`: aggregate metrics and paired CIs;
- `results/audit.json`: matching, partition, evidence-boundary, canonical-data,
  and exact emitted-file hashes;
- `results/optimization_effect.png`: rendered comparison.
- `results/engine_smoke.json`: real SGLSim Engine/DP/Scheduler topology smoke;
  four requests complete and produce eight nonzero committed-prefix updates,
  advancing each protected request from 4,096 to 4,104 safe tokens.

## Evidence boundary

This is a deterministic, synthetic, metadata-level SGLSim experiment using the
timing coefficients in `config.json`. It validates policy mechanisms and
quantifies their modeled effect; it is **not** physical Mooncake/SGLang runtime
evidence, a hardware calibration, or a paper-ready end-to-end result. Transfer
service is reported separately from serving wall-clock time because SGLSim's
Pipeline v2 overlaps communication bookkeeping with forward execution.
