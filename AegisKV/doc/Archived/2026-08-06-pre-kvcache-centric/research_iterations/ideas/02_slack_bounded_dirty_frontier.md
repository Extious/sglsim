# Idea 2: Slack-Bounded Dirty Frontier

## Claim

Do not checkpoint an entire request at a fixed interval. Bound the amount of
unprotected KV by how many tokens can still be recomputed before Algorithm 1's
recovery deadline.

## Mechanism

For request `i`, maintain three distinct frontiers within one request/topology
generation:

- `K_i` — **materialized-KV frontier**: the latest contiguous KV extent that
  exists as a source for protection;
- `C_i` — **scope-committed KV frontier**: the longest contiguous KV prefix
  whose required logical components, failure-domain data quorum, and durable
  manifest record are committed;
- `E_i` — **emitted-output frontier**: output already exposed to the client,
  bound to sampling/RNG and streaming-offset state.

The dirty suffix is

`dirty_i = K_i - C_i`.

`E_i` is not substituted for `K_i`. In ordinary autoregressive decode the last
emitted token may not yet have a KV page, and speculative decoding can widen
that difference. Recovery must neither re-emit `E_i` nor claim that it is
protected by `C_i`.

Algorithm 1 supplies conservative **failure-time recovery budget** `S_i`. It is
not the deadline of a copy started now. The runtime measures
or predicts:

- failure detection and reroute budget `D_i`;
- restore/setup budget `R_i`;
- conservative recompute throughput `q_i` tokens/s for the relevant
  TP/PP/DP/EP topology.

The maximum safely replayable suffix is constrained by the RTO condition

`D_i + R_i + T_recompute(dirty_i) <= S_i`.

With a conservative token-rate approximation, that gives

`U_i = floor(q_i * max(0, S_i - D_i - R_i))`.

If the materialized-KV frontier grows at conservative rate `g_i` tokens/s, the
current protection job's derived expiry is approximately

`T_due_i = (U_i - dirty_i) / g_i`.

This is when the recovery envelope would become unsafe if no new extent
commits. Algorithm 2 does not equate `S_i` with `T_due_i`, and it does not need
a complicated score:

1. if `dirty_i <= U_i`, defer protection;
2. before the derived expiry, create page-aligned protection jobs that keep
   `dirty_i <= U_i`;
3. order jobs by protection deadline, then criticality class;
4. stop only at an extent boundary and advance `C_i` only after the logical
   component quorum and scope-durable manifest ACK; advance `E_i` through a
   separate output/stream publication path;
5. require both a write-path plan and a failure-time restore/recompute plan
   with an available replacement compute target;
6. if neither recovery plan can satisfy `S_i`, publish an explicit
   `RECOVERY_SLO_INFEASIBLE` risk and invoke admission/reroute/escalation rather
   than labeling recompute as an acceptable decision.

## Safety and robustness

`q_i` is a lower confidence bound, not an average. Prediction error changes
how much work is protected, but cannot publish a false checkpoint because the
scope-durable, group-ACKed prefix invariant is independent.

The plan is a dual-path certificate:

1. a normal-time path that can materialize and commit the extent before
   `T_due_i`;
2. a degraded-topology path containing a replacement/warm GPU, model and
   communicator readiness, remote read/H2D or parity decode, plus bounded
   suffix recomputation.

A successful write without the second path is stored state, not a
recovery-SLO guarantee.

The deadline is recalculated when:

- the request enters a new workflow stage;
- downstream fan-out or tool latency changes;
- the topology epoch changes;
- queueing makes restore or recompute slower;
- a newer committed prefix supersedes an older one.

If `T_due_i <= 0`, the request is already outside its protected recovery
envelope. It is an immediate escalation/infeasibility condition, not a
negative-duration copy deadline.

Hysteresis uses two thresholds: start protection at `dirty > U_high`, stop at
`dirty < U_low`. This prevents per-token start/stop oscillation.

## Why it is useful for multi-agent workflows

Agent branches have different downstream dependence:

- a speculative branch that can be discarded receives a large replay budget;
- a branch feeding many children receives a small dirty budget;
- a tool-wait interval can create a protection window without changing the
  generation critical path;
- a parent prefix shared by many descendants should have a long protection
  lifetime even after the parent request completes.

Algorithm 1 represents those semantics only as a conservative deadline and
criticality class. Algorithm 2 remains a lower-level resource mechanism.

## Evidence

- GhostServe checkpoints streaming KV in chunks and reports that hybrid
  restore/recompute is needed in some regimes (physical pp. 5--8 and 11).
- HCache shows that exact KV can be regenerated from smaller intermediate
  state and that storage versus recompute is a legitimate systems tradeoff
  (physical pp. 1, 6, 8).
- The live SGLang backup path already advances completion by stride-aligned
  extents and currently uses a physical-TP-rank minimum. The proposed commit
  instead evaluates required logical attention shards and treats replicated
  copies of one logical head as alternatives; see
  [live storage audit](evidence/live_sglang_storage_audit.md).
- The profiler found dense prefill steps and fragmented decode gaps. A
  page-aligned dirty frontier permits small, revocable jobs instead of an
  all-or-nothing copy; see
  [profiler report](evidence/sglang_profiler/REPORT.md).

## Falsifiable experiment

Compare fixed-period full checkpoints, fixed-token chunking, and the
slack-bounded frontier under heavy-tailed agent DAGs.

Vary slack-prediction error by `-50% ... +100%` and inject GPU/device, process,
and node failures, separating Scheduler-owned Host memory from
process-external Host-pool ownership. Report:

- `K_i`, `C_i`, and `E_i` divergence at failure;
- scope-committed coverage and manifest-quorum availability;
- recovery WCT and recomputed tokens;
- copied bytes per saved recompute token;
- protection-deadline miss rate;
- no-failure P99 and controller oscillation.

The idea fails if conservative `q_i` makes almost every request continuously
protect, if deadline changes cause excessive churn, or if metadata/ACK cost
dominates page-sized extents.

## Novelty boundary

The novelty is not “checkpoint periodically” or “use slack as a weight.” It is
the coupling of a request recovery deadline to separate materialized-KV,
scope-committed-KV, and emitted-output frontiers, a measured replay envelope,
and incremental protection jobs whose correctness is data-and-manifest
ACK-based.
