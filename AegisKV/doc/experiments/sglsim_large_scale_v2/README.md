# SGLSim large-scale AegisKV campaign v2

This directory is an overnight, simulation-only evaluation campaign for the
current `source_owned_independent_hbm_dram_v2` design.  It is deliberately
separate from the V100/A100 Mooncake producer/consumer chain: physical runs are
not required, and no result from this directory may be labelled measured
hardware evidence.

The campaign replays the pinned 393-workflow Weka corpus (98,827 requests) and
varies workload concurrency, model/parallel layout, worker-failure rate, and
simulated cluster scale.  Every policy in a cell receives the same workload,
failure schedule, and random seed.  The main comparison includes the existing
SGLSim baseline adaptations and three leave-one-algorithm-out AegisKV variants;
the central scenario additionally runs all eight Algorithm-1/2/3 on/off
combinations.

## Evidence contract

- Protocol evidence class:
  `sglsim_metadata_des_v2_no_standby_rotation_conservative_sensitivity`;
  every raw cell is explicitly labelled `simulated_sglsim_only`.
- Timing, bandwidth, and capacity are explicit simulation inputs, not measured
  values.
- The source trace SHA-256 and resolved protocol are recorded before execution.
- Raw results are immutable per campaign ID; a rerun requires a new campaign
  ID.
- A failed execution writes an immutable `attempt-*-failed.json`, including
  completed raw hashes and failure tracebacks, but never claims the canonical
  `campaign-index.json`. Retrying the same selection validates and reuses those
  raw cells, then reruns missing or failed cells. Only a zero-failure
  completion commits the canonical index.
- Superiority is never a validity gate.  Negative or neutral results remain in
  the aggregate and paper inputs.
- Headline A/B/C confidence intervals use only an outer bootstrap over the ten
  paired seed statistics.  A seed statistic first averages all 393 workflows
  for A or all 64 equal-exposure probe strata for B/C.  The retained
  workflow-level hierarchical resampling is a within-trace sensitivity only;
  it is not a headline confidence interval.  Duplicating a trace epoch does
  not create a new independent sample.
- The primary failure metric is a fail-now counterfactual,
  policy-independent actual-Slack harm surrogate.  Its label is derived from
  actual durations at a request/tool boundary and decremented by elapsed time
  at the fault.  It does not recompute the concurrently evolving residual DAG
  at fault time, is not end-to-end workflow re-execution, and must not be
  abbreviated to measured WCT.  Internal `ground_truth` gate names mean only
  that this offline label is independent of the policy predictor.

## Current-design alpha interpretation

Algorithm 3 is pressure/stability driven.  It does **not** read the injected
failure rate directly.  Formal validity checks lock the configured driver,
alpha bounds, transition equations and preconditions, pause-before-reclaim
ordering, and independent HBM/DRAM capacity accounting.  They do not require a
particular treatment effect.  The following are preregistered descriptive
diagnostics and are retained even when null: load versus pressure/alpha,
whether stable low-pressure windows exercise additive increase, whether
high-pressure windows exercise multiplicative decrease, and the observed
effective-alpha-zero window fraction.  Alpha zero is structurally permitted
by quota quantization but is neither forced nor a validity/superiority gate.
Failure-rate sweeps report conditional alpha correlation and never require
monotonicity.

The `t=0` control row is an initial telemetry snapshot only: alpha remains at
0.5, the stable-window counter remains zero, and no AIMD transition is
applied.  Only positive-time 30-second periodic windows may apply Algorithm 3;
the C=224 mechanism-horizon check therefore counts six actual opportunities
through 180 seconds rather than treating initialization as an update.

Algorithm 1 has a separate input/wiring identifiability gate: the central
modeled-service cross-fit must yield nonzero predicted policy slack for a
preregistered request fraction, while the no-Algorithm-1 path must consume
exactly zero policy slack.  Same-seed full (`111`) versus no-Algorithm-1
(`011`) admission/copy/resource differences are reported as post-treatment
diagnostics only.  An observationally identical ablation is a valid null
result and is never rejected.

The compositional ablation is a hierarchical structural 2^3 closure, not
three orthogonal independent switches.  When Algorithm 2 is disabled, the
adapter follows StaticIndependent admission and does not consume Slack;
therefore `101` is structurally equivalent to `001`, `100` to `000`, and the
primary `101` path to static FCFS.  Algorithm 1 effects are interpretable only
conditional on Algorithm 2 being enabled.  The equivalent cells remain in the
preregistered cube and are reported as structural/null results, never treated
as a validity failure.

For every scenario, the workload projector now recomputes exact actual suffix
remaining work and boundary CPM Slack from that scenario's realized service
vector.  The policy path remains causal: it scales the workflow-disjoint
central cross-fit remaining prediction using only the current request's
projected prompt/reuse, declared model coefficients, and effective requested
output cap, then applies CPM at the complete currently visible frontier.
Realized output and actual suffix truth are endpoint-only.  The error
multiplier is applied after scenario CPM; `m=1` directly selects the base
prediction without reading actual Slack.  The base prediction, effective
policy input, actual label, visible frontier, and remaining/Slack arrays are
separately hash-bound.

This fail-now model does not execute queue outage, rescheduling, or reduced
primary offered/service load after a failure, so it cannot validate the
hypothesis that more failures indirectly reduce active primary load.  The
failure-rate axis reports the observed pressure/alpha trajectory only and does
not infer that unmodeled mechanism.

Failure-history evidence uses two non-interchangeable snapshots. The top-level
`failure_state_at_horizon` freezes current outages after all events at or
before the base run horizon. `final_state.failure_history` is explicitly the
post-drain state after scheduled repairs beyond that horizon. Reports bind
`*_at_horizon` only to the former and `*_post_all_events` only to the latter;
ever-failed and cumulative old-copy invalidation counters must agree across
both snapshots.

## Planned matrix

The frozen protocol generates 33 unique scenarios, 311 policy-scenarios, and
3,110 paired policy-seed runs:

- 16 concurrency x failure-rate scenarios;
- three additional DP/cluster-scale scenarios;
- one additional Qwen3-32B PP=4 scenario;
- three additional context-scale scenarios;
- three additional effective-network-bandwidth scenarios;
- two additional Peer-HBM-quota and two additional Store-DRAM-quota scenarios;
- three Algorithm-1 prediction-error sensitivity scenarios (full AegisKV
  only);
- ten primary policies per scenario and four additional compositional
  Algorithm-1/2/3 policies at both the central-rate and matched rate-zero
  normal-state scenarios, so the full 2^3 ablation does not inherit prior
  Poisson-fault state;
- ten paired seeds.

The runner retains all 393 workflow rows, all control-window rows, all failure
rows, and all repair rows in immutable gzip artifacts. Request-level retention is a
deterministic sample plus every failed request. This permits paired
full-workflow recomputation and full telemetry re-audit without manufacturing
independence or multi-gigabyte request CSVs. Any hierarchical workflow
bootstrap is labelled sensitivity-only.

Every run also retains all 64 policy-independent active-and-sealed source
probes.  The report keeps three distinct estimands: (A) the full-horizon
uptime-Poisson per-workflow max-overrun join-harm surrogate, computed as each
workflow's maximum failed-request actual-Slack overrun and averaged over all
393 workflows, with idle/ineffective events retained as zero; (B)
active-and-sealed fault-conditioned recovery over 64 equal-exposure strata;
and (C) the single-fault normal-state eligible hazard-integrated additive
burden, which multiplies an exact matching rate-zero B reference by the target
hazard. A is not an event-additive sum and is not executed/replayed workflow
WCT. C is a first-order partial additive counterfactual, not WCT, MTBF, total
burden, or a steady-state observation, and it is never added to A.
Nonzero-rate B preserves the rate-local live invalidation/repair history;
rate-zero B and C are failure-free normal-state references.

The legacy scenario fields `hbm_quota_gib_per_target` and
`dram_quota_gib_per_target` denote per-physical-GPU/stage-shard quotas. A
logical DP target therefore has aggregate capacity equal to the per-shard
quota times `PP x TP`; reports include both values so the Qwen3-32B PP=4 point
is not mislabeled as having only one shard's capacity.

## Commands

```bash
uv run python AegisKV/experiments/sglsim_large_scale_v2/run_campaign.py --pilot
uv run python AegisKV/experiments/sglsim_large_scale_v2/run_campaign.py
uv run python AegisKV/experiments/sglsim_large_scale_v2/analyze.py --input-dir <lineage>/full
uv run python AegisKV/experiments/sglsim_large_scale_v2/audit_campaign.py --campaign-dir <lineage>/full --analysis-dir <analysis> --require-full --output <analysis>/prefigure_full_audit.json
uv run python AegisKV/experiments/sglsim_large_scale_v2/plot_results.py --analysis-dir <analysis> --prefigure-audit <analysis>/prefigure_full_audit.json
uv run python AegisKV/experiments/sglsim_large_scale_v2/audit_campaign.py --campaign-dir <lineage>/full --analysis-dir <analysis> --figures-dir <analysis>/figures --require-full --output <analysis>/final_audit.json
```

Formal figures fail closed unless they hash-bind a canonical PASS from the
independent full raw+analysis audit.  A self-declared `COMPLETE` analysis is
insufficient.  The final audit then verifies the figures and their prefigure
audit binding.  Audit outputs are immutable: an existing path is never
overwritten with different content.

The default pilot is an exact nine-run diagnostic: one paired seed, the
rate-zero, central-rate, and highest-rate central scenarios, and policies
`111`, `011`, and recompute-only.  It validates the K64 schedule, the matching
rate-zero reference, and mechanism/accounting contracts; it is never a paper
number source.  The pilot must complete and report empirical wall time before
the full worker pool is launched.

The offered-concurrency axis uses a seeded, policy-independent projected
trace-lifespan schedule; it is not a closed loop over simulator FCFS or
failure-adjusted completion, and it does not reuse the raw corpus's
917-million-ms wall-clock span. Each workflow's
relative request offsets are scaled by the ratio of median simulated request
service to median raw API time. The offered schedule launches `min(C,N)`
workflows at time zero and one pending workflow at each projected trace-lifespan
completion. The steady controller-analysis interval begins after a
40-projected-completion stabilization prefix and ends at the last launch;
drain is reported separately. All 393 workflows, including the prefix and
drain, remain in full-workflow harm/resource accounting and base-timing
diagnostics; no end-to-end failure WCT is claimed. C=224 was selected before
any performance result solely for mechanism observability: across all ten
projected schedules, the last launch is at or after 180 seconds, exposing at
least six periodic AIMD update opportunities and at least five tagged steady
samples.  Alpha zero and performance advantage remain diagnostics, not gates.
Poisson
worker-shard fail-now counterfactuals cover the complete total-GPU-shard
exposure horizon (DP x PP x TP processes) and retain idle/ineffective events. A failure estimates
recovery for that rank's active source request and conservatively invalidates
the affected logical PP replica's aggregate Peer-HBM protection (not a
stage-local partial loss); the request's separately bound peer target is not
failed by that same event, but its old Peer-HBM copy may already be unavailable
after an earlier uptime event. DRAM copies survive, with no re-protection.
Physical failure history is keyed by `(logical source rank, TP/PP shard rank)`;
current/ever-failed logical replica-target outage and cumulative old-copy
invalidation are separate conservative aggregate counters. Different PP=4
stage-shard failures are therefore not mislabeled as a repeated physical
failure merely because they share a logical source rank.

The authoritative current design and source implementation require atomic
rotation to a healthy compatible standby when a bound target fails, followed
by HBM admission of subsequent chunks against the replacement target's real
pool.  This large-scale DES does **not** migrate its per-target controller,
pending-copy, and independent-DRAM state, so it deliberately does not model
that rotation: `standby_rotation_modeled=false` and
`complete_current_design_target_failover_evaluated=false`. Natural prior-target
invalidation/repair is retained only as a conservative no-standby-rotation
sensitivity, never as evidence for complete current-design failover.
Pressure lasts
until an explicit repair event at t+60 seconds, which re-enables only future
HBM admission; invalidated old copies remain invalid. This simulator does not
execute worker replacement, queue outage, or full fail-stop rescheduling. The
time-weighted offered-schedule mean must be within 5% of the target.
Every artifact separately recomputes realized base and failure-adjusted active
workflow mean, P50, P95, and peak from workflow arrival-to-join intervals.

The Qwen3-8B and Qwen3-32B setup/prefill/decode coefficients and the 1/10/20/40
GB/s effective-path sweep are preregistered synthetic inputs, not physical
calibration.  The 10 GB/s center is 100GbE-class; 1/20/40 GB/s are sensitivity
points. PP=4 uses four independent stage lanes in aggregate; shared-NIC
contention is not modeled.

The retained producer key `wct_measurement_workflows` is a legacy compatibility
name for the exact 393-workflow aggregate count.  No user-facing metric derived
from it is a measured WCT.
