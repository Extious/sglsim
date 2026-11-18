# G0/G1 opportunity and decision-falsification harness

## Purpose

This directory is reserved for two empirical questions that the analytical
results cannot answer:

- **G0 decision relevance:** on held-out traces, does the current Aegis policy
  select different actions from matched load/size, criticality, TTL, and
  shortest-recovery policies often enough to matter?
- **G1 opportunity:** how often do causal `BOUNDED` deadlines, action crossings,
  and EDF order changes exist, and how many workflows/KV bytes do they cover?

G0/G1 do **not** need a GPU fault experiment.  They establish that the new
decision signal is present and non-vacuous.  Whether a different decision
actually improves workflow WCT/SLO is a separate G4 paired hard-fault question.

The current semantic reference is deliberately split into three small modules.
`python/sglsim/srt/simulation/aegis_prefix_recovery.py` derives each job's
multi-tier service by a shortest-path DP over continuous KV-group boundaries;
an action edge is invisible before commit and appears after failure only when
pin/read-lease validation succeeds.  Normal protection expiry remains a
separate portfolio-horizon check.
`python/sglsim/srt/simulation/aegis_recovery_laxity.py` implements one
Router-fixed failure domain with absolute recovery deadlines, target-local EDF,
workflow-frontier `max`, and lexicographic per-domain risk.
`python/sglsim/srt/simulation/aegis_protection_portfolio.py` composes those
scenarios under the cross-domain robust pair, typed multi-resource budgets,
prefix dependencies, dominant-share admission, deterministic victim cover, and
full-exchange re-scoring.  Candidate gain is evaluated at its conservative
commit horizon on the original absolute deadline axis; an action that cannot
finish before its validity bound is refused.  Neither module is in the online
SGLang runtime or evidence of physical KV movement.  The older
`aegis_workflow_harm.py` jointly searches target assignments, bystander queues,
and weighted workflow harm; it is retained only as a full-`J_d`/legacy oracle
diagnostic and is **not** the production Algorithm 2.

## Claims that are not experiment gates

The following belong to the theory appendix, not this harness:

- positive hard Slack is unidentifiable without a revealed continuation bound;
- `s^B=[L_{-i}-U_i]^+` is a conservative absolute deadline witness and cannot
  be recharged at failure time;
- the prefix shortest-path recurrence and monotonicity, plus the three-region
  single-job action value and its ordering counterexamples;
- a completed action has value only after group commit/pin visibility;
- ideal EDF minimizes maximum lateness under its stated single-lane premises;
- exact protection selection is NP-hard by reduction from 0--1 knapsack;
- the strict chunk executor has the stated one-quantum additive bound when all
  of its premises hold;
- the HBM capacity ledger invariant follows by induction.

Trace replay may illustrate these statements or measure how often their
premises arise, but it must not be presented as proof.

## Frozen online comparison contract

Every policy must receive the same:

- revealed workflow state and ticket provenance available at decision time;
- Router-fixed standby target and route version;
- committed continuous prefix and layout compatibility;
- candidate actions, completion horizon, multi-resource demand, and hard caps;
- deployment-specific conservative target-local recovery service bounds;
- source-domain contingency set and failure timestamp;
- Store/peer-HBM budget and transfer envelope.

For each declared single-source-domain fail-now scenario `d`, evaluate the
current fixed-target EDF policy with

```text
R_d(S) = (maximum deadline tardiness,
          sum of per-workflow frontier overrun).
```

Normal-time protection uses the parameter-free robust pair

```text
R(S) = (max_d R_d.maximum_tardiness,
        sum_d R_d.workflow_overrun).
```

The second component is a deterministic tie-break, not expected loss.  Do not
fit tenant weights, source-domain probabilities, or a CVaR coefficient from the
evaluation trace.  Operator isolation, if required, is enforced by separate
quota/risk-class pools outside this objective.

The minimum baseline set is:

| Policy | Allowed decision signal |
|---|---|
| `load_size` | recoverable work/bytes and target load only |
| `criticality` | the same actions plus an online request criticality signal |
| `ttl` | return/wait estimate and tier benefit under the same capacity budget |
| `shortest_recovery` | smallest protected recovery service first |
| `aegis_deadline` | current causal deadline risk and fixed-target EDF |

Signal ablations are useful for debugging, but final related-work comparisons
must use author code or an independently audited faithful implementation with
the same action space, resource budget, copy bandwidth, and failure schedule.

## Pre-registered G0/G1 outputs

The held-out study must report, with workflow-disjoint confidence intervals:

1. fraction of workflows and KV bytes with valid `BOUNDED`, `UNKNOWN`, and
   `DEFERRED` tickets;
2. fraction satisfying `p_protected < deadline < p_unprotected`;
3. fraction whose action changes target-local EDF dispatch/preemption order;
4. pairwise action-set disagreement against every matched baseline;
5. disagreement conditioned on model, DAG shape, failure position, KV size,
   load, and source domain;
6. selected-action completion-before-expiry and failure-before-commit rates;
7. current heuristic objective and exact small-instance oracle gap, with
   `status`, objective, bound, gap, solve time, and instance hash.  The checked-
   in exhaustive certificate covers a static snapshot of already-committed
   actions and refuses nonzero copy lead times.  Issuance/order regret with
   in-flight actions belongs to the separate time-expanded/full-`J_d` oracle;
   the two certificates must not be conflated.

Before opening the formal test split, freeze the stop/go thresholds from the
main design: among workflows with an application-declared mandatory fan-out/
fan-in region, the clustered 95% lower confidence bounds must reach 30% for
workflows with at least one positive `BOUNDED` deadline and 20% for covered
active KV bytes; crossing, EDF-order change, and strongest-baseline action-set
disagreement each require a 10% lower bound.  Unconditional all-workload
coverage and per-template/model/layout strata are mandatory, so the eligible
denominator cannot hide a narrow scope.  These are research continuation gates,
not universal guarantees or proof obligations.

A constructed counterexample is sufficient to prove non-equivalence of two
decision functions, but it cannot pass G0.  Conversely, G0/G1 do not establish
realized WCT benefit; that belongs to G4.

## Real-trace evidence boundary

The checked-in Real-CC artifacts contain workflow/fan-out identities, branch
durations, failure-position counterfactuals, and Algorithm-1 snapshots.  They do
not contain all of the following current online inputs:

- source failure-domain placement;
- complete residual scope or mandatory-join region, completion barrier,
  graph/bound/service-envelope versions, a fenced witness-contract ID and
  validity horizon, and per-node hard duration witnesses;
- Router-fixed compatible target and route version;
- causal target availability/release times;
- matching group-committed continuous prefix and action completion horizon;
- deployment-specific conservative target-local load/DRAM/SSD/recompute bounds;
- layout compatibility and source/object lifetime witnesses;
- `F_serve`/`F_recover` resource envelopes and per-action resource demand.

Missing fields must yield `UNSUPPORTED_BY_SCHEMA` or `REFUSED_TO_SCORE`; they
must not be filled with hidden constants.  Until an enriched held-out trace or
instrumented serving run records these fields, G0 and G1 remain `NOT PROVEN`.

`python/sglsim/srt/simulation/aegis_bounded_slack.py` now provides the
deterministic scoring path once such a trace supplies either a complete
versioned residual DAG or a closed region ending at the next mandatory
AND-join, plus hard lower/upper duration and service-envelope witnesses.  A
closed region must record its barrier and every node must reach it.  One fenced
contract must keep the scope and reserved service valid through its upper
completion horizon; best-effort invalidation after a weakening event is not a
hard witness.  The helper preserves the original source/sink sets when
computing paths and fails the whole witness closed on incomplete, missing,
stale, or prematurely expiring input.  Its constructed property tests
establish formula conformance only; they do not supply the missing real-
workload fields or pass G1.

## Strict bounded-opportunity input and scorer

`bounded_scope.schema.json` is the version-3 sampling-frame contract for the
deterministic path.  Every decision selected by the frozen `sampling_policy_id`
must appear, including decisions with no usable proof scope.  Such a row is a
valid `UNAVAILABLE` zero-opportunity observation with an explicit reason and
its total failure-exposed active KV bytes; it is not silently dropped from the
denominator.  Nodes outside the declared failure domain remain in the DAG so
their duration can witness Slack, but carry zero `failure_exposed_kv_bytes` and
cannot create a protection opportunity for that scenario.
`score_bounded_opportunity.py` deliberately implements the same contract
without permissive coercions: unknown fields, missing fields, duplicate
decision identities, string-valued booleans, non-finite numbers,
future-information provenance, and formal rows without a workflow-disjoint
split become `UNSUPPORTED_BY_SCHEMA`.  It never fills a graph, bound, KV size,
layout, or failure domain from a default.

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=python uv run python \
  AegisKV/experiments/g0_g1/score_bounded_opportunity.py \
  --input-jsonl /path/to/frozen-bounded-scope.jsonl \
  --output-json /path/to/bounded-opportunity-points.json
```

The output keeps `synthetic_development` and `held_out_formal` summaries
separate, reports both unconditional and declared-parallel-region coverage,
and always emits `gate_assessment=NOT_ASSESSED`.  The KV-byte fraction is
explicitly a decision-snapshot-weighted point estimate; repeated decisions
from one workflow are not IID observations.  Passing G1 additionally requires
a sampling cadence frozen before opening the test split, workflow-clustered
confidence intervals, and the preregistered template/model/layout strata.
Those inference steps must consume the scored rows and cannot relabel a schema
rejection as zero or silently drop it.

`assess_bounded_opportunity.py` implements only the next, deliberately narrow
inference step.  It refuses any schema rejection, development row, mixed
collection contract, unfrozen preregistration, zero denominator, or contract
mismatch.  It uses one workflow-level binary observation for the positive-
`BOUNDED` workflow fraction (Wilson interval) and resamples complete workflows
for the decision-snapshot KV-byte ratio.  Repeated snapshots from one workflow
therefore cannot manufacture a large sample size.  The checked-in
`bounded_opportunity_preregistration.template.json` has
`frozen_before_test_open=false` on purpose: it is a template, not evidence.
Before opening a new workflow-disjoint formal split, copy it, bind the exact
dataset/split/sampling-policy IDs, choose the minimum sample counts and seed,
record its hash externally, and only then set the flag true.

```bash
PYTHONPATH=python .venv/bin/python \
  AegisKV/experiments/g0_g1/assess_bounded_opportunity.py \
  --scored-json /path/to/bounded-opportunity-points.json \
  --preregistration-json /path/to/frozen-preregistration.json \
  --output-json /path/to/bounded-opportunity-assessment.json
```

Even `OPPORTUNITY_SUBGATE_PASS` is not a G1 pass.  The v3 bounded-scope schema
does not contain protected-vs-unprotected action crossing or EDF dispatch/
preemption change, and it does not assess G0 matched-baseline disagreement.
The assessor therefore emits
`NOT_ASSESSED_ACTIONABILITY_AND_EDF_CHANGE_MISSING` until a separate frozen
decision-study artifact supplies those obligations.  It also says nothing
about realized WCT/SLO benefit, which remains G4.

## Status of the checked-in script

`run_counterfactual.py` still exercises the legacy full-`J_d` tiny model and
produces only `synthetic_semantic_check` and `schema_capability_audit` artifacts.
It is useful for regression tests and offline-oracle bookkeeping, but it cannot
be cited as evidence for the current robust fixed-target Algorithm 2 or for
real-workload prevalence.

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=python uv run python \
  AegisKV/experiments/g0_g1/run_counterfactual.py \
  --output-dir AegisKV/experiments/g0_g1/results
```

All generated JSON/CSV files must retain their `evidence_class` field.
