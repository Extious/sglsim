# Exact finite-horizon optimizer for Algorithms 2 and 3

Status: mathematical implementation contract, 2026-08-04.

The claim is deliberately narrow: the solver may certify a global optimum for
the finite, discretized problem below (or a predeclared rolling-horizon
instance of it).  It is not a claim of universal optimality for an arbitrary
online system.

## Finite problem

For one optimization window, define:

- requests `r in R`, workflows `w in W`, PP stages `p in P`, ordered KV chunks
  `c in C_r`, eligible backup DPs `d in B_r`, tiers
  `T={HBM, HOST, RECOMPUTE}`, nodes `n`, links `l`, and failure/recovery
  scenarios `omega in Omega`;
- immutable execution DP `e_r`, workflow DAG, chunk token/byte sizes, SLO and
  trace-oracle Slack, stage capacities and layer ranges, link bandwidth, node
  mapping and failure domains;
- calibrated normal-path interference, copy, restore, recomputation, queueing,
  and budget-switching coefficients;
- scenario probability `pi_omega`, failed domains, arrival/completion events,
  and service-time samples.  Scenario input may be empirical or sampled from a
  frozen distribution, but never silently replaced by a deterministic mean.
- causal admission states `a in A`, each with a half-open logical-time interval
  and the DP ranks already known unavailable when Router freezes the route.

Trace-oracle Slack uses future Weka API completion information.  It is an upper
bound input with Algorithm 1 bypassed, not an online Algorithm 1 result.

## Decision variables

- `y[r,d] in {0,1}`: immutable backup DP choice.
- `x[r,p,c,d,t] in {0,1}`: place the stage-local chunk in HBM, Host, or mark it
  for recomputation at backup DP `d`.
- `q[r,p,c,d,omega] in {0,1}`: that placement survives scenario `omega`.
- `g[r,c,d,omega] in {0,1}`: chunk `c` is committed and recoverable at every
  required PP stage under scenario `omega`.
- `H[d,k]` and `M[d,k]`: integer HBM-per-stage and Host policy budgets for DP
  `d` in window `k`; published alpha/beta are exact normalized views of these
  integers.
- `DeltaH[d,k]`, `DeltaM[d,k]`: absolute budget changes for switching cost.
- node/link admission and occupancy variables for physical arbitration.
- per-stage restore/transfer/recompute completion variables, request completion
  `C[r,omega]`, workflow completion `Cw[w,omega]`, SLO loss
  `L[w,omega]`, and CVaR variables `eta`, `xi[omega]`.

## Core constraints

### Routing and exclusive placement

```text
sum_d y[r,d] = 1
y[r,e_r] = 0
sum_t x[r,p,c,d,t] = y[r,d]
```

Eligibility masks set `y[r,d]=0` when a failure-domain or topology constraint
forbids that pair.  Every chunk has an explicit HBM, Host, or Recompute outcome.

Admission availability is not a failure prior.  If request `r` is admitted in
observed state `a(r)`, the exact model additionally enforces:

```text
y[r,d] = 0                       for d in unavailable(a(r))
e_r not in unavailable(a(r))
arrival_r in [state_start, state_end)
```

Every chunk of a request carries the same admission state and time.  The state,
unavailable-rank set, and information mode are included in the instance and
runtime wire contract, independently revalidated after solving, and emitted in
the Router event.  The runtime fails closed if current availability differs
from the state for which a static route was certified; a causal rolling solve
must publish a newer plan instead of silently falling back to a heuristic.

An admission state may be labeled `causal_observed`, `topology_robust`, or
`future_manifest_oracle`.  The last label permits a diagnostic upper bound but
is excluded from deployed and completion-gate evidence.  In particular, the
benchmark's realized failure manifest never overwrites the separately frozen
failure-hazard scenario prior.

### Stage alignment and prefix commit

Only variables whose source and destination share `pp_rank=p` are created.
For protected tiers define `u = x_HBM + x_HOST`.  The all-stage AND is encoded
exactly for every `(r,c,d,omega)`:

```text
q[r,p,c,d,omega] = surviving protected placement at stage p
g <= q[r,p,c,d,omega]                         for all p
g >= sum_p q[r,p,c,d,omega] - (|P|-1)
g[r,c+1,d,omega] <= g[r,c,d,omega]            contiguous prefix
```

Survival coefficients are fixed by the scenario's failed domains and selected
tier.  Thus no prefix is recoverable unless all four stage-local chunks are
committed, version-matched, and surviving.

### DP-owned budgets and node arbitration

For every DP `d`, stage `p`, and window `k`:

```text
sum chunk_tokens * x_HBM <= H[d,k]
0 <= H[d,k] <= stage_hbm_protection_cap[d,p,k]
alpha[d,k] = H[d,k] / declared_stage_reference_capacity[d,k]
```

The same integer `H[d,k]` (or the same alpha multiplied by each declared stage
capacity) constrains all four PP stages.  For Host:

```text
sum host_tokens attributed to d <= M[d,k]                 DP beta policy
sum host_bytes resident on node n <= host_capacity[n,k]   physical capacity
sum copy_bytes on link l in interval j <= bandwidth[l,j]
```

Both layers remain explicit even when the current placement maps one DP to one
node.  Absolute budget changes are linearized in the standard way, e.g.
`DeltaH >= H[k]-H[k-1]` and `DeltaH >= H[k-1]-H[k]`.

### Recovery and workflow loss

Per-stage recovery time includes the selected surviving tier's transfer time,
queue delay, and explicit recomputation for the uncovered suffix.  Request
recovery completion is at least every stage completion, so it is the pipeline
bottleneck.  Workflow completion obeys the original DAG precedence and is at
least every terminal request completion.

```text
L[w,omega] >= Cw[w,omega] - workflow_slo[w]
L[w,omega] >= 0
xi[omega] >= total_loss[omega] - eta
xi[omega] >= 0
```

Normal execution, failure interruption, retry admission, and pipeline service
are included in `C`; recovery benefit is therefore expressed directly as WCT
or SLO loss rather than a clipped deterministic proxy value.

## Objective

The default registered objective is:

```text
min  E_omega[
       sum_w (lambda_wct * Cw[w,omega] + lambda_slo * L[w,omega])
     ]
   + lambda_cvar * (eta + E_omega[xi[omega]] / (1-tau))
   + normal_path_interference_cost
   + hbm_memory_cost + host_memory_cost + copy_bandwidth_cost
   + alpha_beta_switching_cost
```

All coefficients, units, scenario probabilities, and `tau` are serialized in
the instance manifest.  Training/tuning may choose among preregistered
coefficient grids; the held-out run uses one frozen selection.

## Exactness and certificates

The initial implementation uses `scipy.optimize.milp` with the bundled HiGHS
backend, which is available in the repository virtual environment.  Every solve
persists:

```text
instance_sha256, solver/version/options, status/message,
objective, dual_bound, mip_gap, node_count, solve_time_s,
decision_sha256, deterministic_seed
```

The result may be labeled **global optimum** only when all of the following are
true:

1. solver status is `OPTIMAL` (`scipy` status 0);
2. the persisted primal solution passes every independently recomputed
   capacity, bandwidth, failure-domain, routing, and stage-prefix invariant;
3. the reported absolute/relative gap is zero within the registered numerical
   tolerance; and
4. a deterministic replay produces the same canonical decision digest and
   objective.

`TIME_LIMIT`, feasible-with-gap, numerical failure, decomposition without a
proof, or heuristic repair is never called globally optimal.  Such a result is
persisted and excluded from exact-oracle claims.

## Scaling boundary

The joint 100-workflow problem may exceed the exact solve budget.  The deployed
Algorithm 2/3 therefore uses a preregistered rolling horizon whose candidate
requests, chunks, windows, and scenarios are fixed before solving each finite
instance.  Each window must still be certified optimal for that finite
instance.  Cross-window state (resident stock, leases, ACKs, budget version,
and switching cost) is an immutable input to the next instance.

No claim is made that independent window optima equal the unrestricted
full-trace optimum unless a separate decomposition proof is supplied.

## Oracle, deployment, and regret

Every diagnostic instance records three decisions on identical input:

- `legacy_greedy`: current value-density placement and feedback heuristic;
- `exact_oracle`: certified optimum for the finite instance, optionally with
  future scenario realization when explicitly labeled diagnostic-only;
- `deployed_exact_rolling`: certified optimum for the deployable rolling
  instance using only the frozen scenario distribution and causal state.

Report objective regret and outcome regret relative to the exact oracle.  A
formula rewrite that still executes the old value-density greedy path does not
count as the new algorithm.

## Required tests

- exhaustive enumeration versus MILP on tiny instances;
- infeasible capacity, bandwidth, and failure-domain cases;
- four-stage prefix AND and monotonic-prefix cases;
- DP beta and node physical-capacity constraints independently;
- objective, dual bound, gap, status, and decision digest persistence;
- deterministic replay under fixed seed and canonical ordering;
- deliberate time-limit/nonzero-gap result rejected by the global-optimum API;
- legacy/exact/deployed regret calculation on the same serialized instance.

## Implementation checkpoint (2026-08-05)

The offline-only foundation is implemented in
`python/sglsim/srt/simulation/aegis_exact_optimizer.py` using SciPy 1.17.1's
HiGHS MILP backend.  It currently covers immutable backup rank, every
request/chunk/PP-stage HBM/Host/Recompute decision, DP HBM/Host budgets, node
Host capacity, directed link bandwidth, failure-domain survival, all-stage
contiguous prefix, scenario workflow loss, empirical CVaR, normal-path/memory/
bandwidth/switching costs, and the certificate fields above.

The finite instance now also represents route-admission time and observed
availability explicitly.  HiGHS fixes route variables for unavailable backup
DPs to zero; construction rejects an unavailable execution DP or an admission
state with no eligible backup.  Independent certificate validation repeats the
availability check.  The solver-free route contract preserves the state and
labels future-manifest knowledge, and the controller rejects a plan/state
mismatch rather than letting a certified plan silently change routes.

The certificate path independently rebuilds routes, placements, capacities,
safe prefixes, workflow losses, and the complete objective.  Six focused tests
originally covered exhaustive enumeration of a tiny problem, failure-domain and
prefix cases, deterministic replay, and rejection of a nonoptimal certificate;
the admission-state tests additionally cover causal candidate masking, interval
validation, unavailable execution/all-backup rejection, wire round-trip, and
runtime fail-closed behavior.

The certified plan is wired through the DP-owned versioned budget publication
and stage-aligned runtime.  The remaining deployment step is event-driven
causal rolling re-solve/publication when availability changes; the bootstrap
artifact is intentionally an all-available static window and cannot claim
optimal deployment for requests newly admitted during an outage.
