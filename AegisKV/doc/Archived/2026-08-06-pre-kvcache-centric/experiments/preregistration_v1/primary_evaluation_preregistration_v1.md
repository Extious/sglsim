# AegisKV primary evaluation preregistration v1

Frozen design date: 2026-08-04.  Status: **design frozen; trace IDs and manifest
hashes must be materialized before any tuning run**.

Changing a frozen field after inspecting AegisKV-versus-No-Backup held-out
outcomes creates a new preregistration version; v1 and all unsuccessful cells
remain retained.

## Research question and completion gate

Under a fair paired Qwen3-32B, DP=4, PP=4, 32K deployment, does the final
deployable AegisKV policy reduce pooled paired mean workflow completion time
relative to No Backup by at least 10% over the complete nonzero-failure primary
matrix?

The point-estimate gate is

```text
100 * (mean(WCT_no_backup) - mean(WCT_aegiskv)) / mean(WCT_no_backup) >= 10.
```

The mean pools every workflow observation in every registered nonzero-failure
held-out cell; each cell has the same workflow count and seed count.  A
stratified paired bootstrap resamples workflow IDs within each
`(load, injected_rate, scope, seed)` stratum for 10,000 replicates.  The 95%
percentile interval is reported and its lower bound must be greater than zero.
The 10% threshold applies to the point estimate; the interval is not silently
substituted with a p99 or retry-latency gate.

Zero-failure cells are a separate normal-path guard: AegisKV mean WCT overhead
relative to No Backup must be no more than 5% when pooled across those cells.
All individual cells are reported whether favorable or not.

## Deployment and evidence boundary

- Model: Qwen3-32B; context length 32,768 tokens.
- Topology: DP=4, PP=4, TP=1; 16 independent Scheduler/worker processes.
- KV transfer: only `source DP.PP_i -> backup DP.PP_i`.
- Online and offline share scheduling, placement, ACK, failure/recovery, retry,
  and metric code.  Offline changes only the clock/driver.
- Primary search/evaluation uses offline logical time only after the registered
  online/offline accuracy gate passes.  Final candidates receive a small
  online wall-clock replay on identical cells.
- Slack source: Weka trace-oracle future API completion.  This bypasses
  Algorithm 1 and is labeled an upper bound in every artifact.
- A diagnostic solver allowed to know the realized future failure manifest is
  an oracle only and is excluded from the completion gate.
- Each deployed rolling solve records the availability state already observed
  at route admission.  A static all-available certificate is invalid for a new
  request admitted during an outage; the causal driver must publish a certified
  state-matched window.  Silent heuristic fallback invalidates the cell.

## Workload split

Source dataset:
`semianalysisai/cc-traces-weka-062126`, whose local audited summary contains
393 workflows and 175 workflows with at least one subagent group.

Selection and split are deterministic:

1. normalize every source workflow using `weka_trace.trace_to_workload`;
2. retain workflows with at least one subagent group, at least one main request
   after the first subagent group, valid nonnegative measured API times, and no
   malformed request rows;
3. sort by `SHA256("aegiskv-primary-v1:" + workflow_id)`;
4. assign the first 25 eligible IDs to smoke/calibration, the next 50 to
   tuning, and the next 100 to held-out primary evaluation.

If fewer than 175 eligible workflows are materialized, v1 fails its input
precondition and no tuning starts; the selection rule is not relaxed after
outcome inspection.  Trace JSONL, ordered ID lists, source digest, request
counts, eligibility rejections, and split digests are persisted.

Every primary cell replays all 100 held-out workflows.  A 5-workflow result is
smoke only and cannot satisfy any gate.

## Frozen primary matrix

- workflow concurrency/load: `2, 4, 8`;
- injected failure rates: `0.5, 2.0` events per 1,000 logical deployment
  seconds for the recovery matrix;
- failure scope: `pp_worker` and `dp_replica`;
- experiment seeds: `901, 1901, 2901`;
- policies: `no_backup`, `legacy_greedy`, `deployed_exact_rolling`.

This yields 3 x 2 x 2 x 3 = 36 paired recovery conditions and 108 policy
cells.  The zero-failure guard adds 3 loads x 3 seeds x 3 policies = 27 cells.
Each policy cell contains 100 workflows.

Failure manifests are generated in logical time before running any policy.
They are reused byte-for-byte across policies.  The policy's failure-hazard
prior is a separately named, tuning-selected value and is frozen before
held-out evaluation; it is never overwritten with the cell's injected rate.
For every `(load, rate, scope)` aggregated across its three seeds, at least 30
unique workflows must be failure-affected.  This exposure check uses only the
manifest and route/workload plan, not A/B outcomes.  A condition that fails the
check is underpowered and cannot enter the gate until a new preregistration
version defines a larger workload or different rate.

The exact realized future failure diagnostic is run only after causal primary
artifacts are sealed, and is never mixed into the primary aggregate.

## Pairing and fairness

Within a condition, every policy receives identical:

- ordered workflow/request rows and arrival times;
- workflow launch gate, concurrency, execution route decisions and immutable
  backup candidate plan (No Backup ignores placement, not routing);
- failure manifest, failed stages, duration, and retry eligibility;
- model/forward/communication calibration and random seeds;
- HBM, Host, link, node, DP, PP, context, and scheduler configuration;
- retry request set, retry trigger, and success/failure definition.

No Backup is not assigned extra failure cost, a weaker router, different
victims, or additional requests.  AegisKV normal-path copy and control overhead
is charged in its own timing/accounting.  Paired-route validation is a hard
cell validity check.

## Tuning boundary

- Smoke/calibration: 25-workflow split; correctness and cost calibration only.
- Tuning: 50-workflow split; choose chunking, objective coefficients, scenario
  count/distribution, rolling horizon, alpha/beta bounds, and policy hazard
  prior from a serialized candidate grid.
- Held-out: 100-workflow split; one frozen final configuration, no
  outcome-driven changes.

All attempted tuning configurations and failed solver/run results are retained.
The held-out result is not cherry-picked across configurations.

## Offline/online accuracy gate

Before formal offline use, identical smoke and full-configuration calibration
runs compare:

- mean/p50/p90/p99 WCT and empirical CDF;
- TTFT/ITL or the available equivalent service-time measures and makespan;
- per-DP queue depth, completion order, route decisions, and per-stage batch
  identity;
- cache, placement, ACK, backup, retry, failure, and recovery event counts and
  causal order;
- forward and communication accounting.

Discrete identities/counts must match exactly.  Continuous tolerances are
registered in the calibration manifest before comparison.  Offline and online
need not be bit-identical in wall-clock timestamps.  A failed gate blocks
formal offline evidence, not merely one chart.

## Required artifacts and metrics

Each cell persists incrementally:

- `experiment_manifest.json`, resolved config, command, code revision, dirty
  diff digest, trace/route/failure hashes, seed and solver options;
- append-only events JSONL and per-stage rank-state JSONL;
- per-workflow and per-retry CSV/JSONL;
- solver instance, objective, bound, gap, status, solve time and decision digest;
- validation report for topology, lifecycle, routing, capacity, bandwidth,
  stage-prefix, pairing, and completion.

Report mean/p50/p90/p99 WCT, SLO misses, retry latency, restored/recomputed
tokens per stage, KV reuse and recompute ratios, normal-path overhead, HBM/Host
occupancy, copy bandwidth, alpha/beta trajectory, evictions, unmet tokens and
utility, and solver diagnostics.

Empirical CDFs use raw per-workflow/per-retry samples with paired/stratified
bootstrap 95% confidence bands and explicit `n`, seed, load, rate, scope and
units.  Editable plotting source plus PNG, SVG and PDF are retained.  Smooth
interpolation is not presented as additional samples.

## Resource guard

Before launching the matrix, record free disk, memory, CPU count, current
processes and predicted bytes per cell from the 25-workflow smoke.  The initial
cap is four concurrent independent cells, one solver thread per cell, isolated
ports/output/temp directories/seeds, and 300 GiB total new artifacts.  Scaling
beyond either cap requires a new recorded resource estimate, not silent shared
state.
