# Stage-aligned AegisKV runtime contract

Status: design frozen before DP x PP implementation, 2026-08-04.

This contract separates immutable routing, DP-owned policy budgets, stage-local
data movement, and node physical-capacity arbitration.  It is shared by the
online wall-clock and offline logical-clock drivers.

## Identities and ownership

- `StageId = (dp_rank, pp_rank, tp_rank)`.  The target deployment is
  `(dp_size, pp_size, tp_size) = (4, 4, 1)`.
- One OS Scheduler/worker process owns each `StageId` and one layer-local KV
  metadata pool.
- `DPReplicaId = dp_rank` owns routing feedback and one versioned
  `DPBudgetSnapshot(alpha_n, beta_n)` shared by PP0-PP3.
- `NodeId` owns only the physical Host allocator and link-capacity ledgers.  It
  does not choose beta and does not publish policy versions.
- The Router owns immutable execution/backup selection.  The Connector cannot
  silently substitute another backup DP.

## Immutable messages

The implementation must use frozen dataclasses (or an equivalent immutable
wire schema) with explicit `schema_version`.

### `RoutingDecision`

Required fields:

```text
request_id, workflow_id, execution_dp_rank, backup_dp_rank,
route_version, decided_logical_time, failure_domain_constraints
```

`execution_dp_rank = e(r)` and `backup_dp_rank = b(r)` are selected exactly
once for a request version.  Every PP stage consumes the same decision.  A
retry creates a new version linked to the prior decision; it never mutates the
old object.

### `DPBudgetSnapshot`

Required fields:

```text
dp_rank, budget_version, window_id, alpha, beta,
hbm_limit_tokens_per_stage, host_policy_limit_tokens,
valid_from_logical_time, objective_id, solver_certificate_id
```

All four stages atomically move from version `v` to `v+1` at a declared window
boundary.  A stage must reject a placement carrying an unknown or stale budget
version.  Alpha is one DP policy ratio applied to each stage's protected HBM
partition; beta is one DP policy limit for total Host leases attributable to
that DP.

### `PPStageBatchEnvelope`

Required fields:

```text
dp_rank, pp_rank, microbatch_id, batch_id, request_ids,
forward_mode, token_frontiers, route_versions, budget_version,
proxy_metadata, causal_sequence
```

External requests enter PP0.  PP0 forwards the envelope to PP1, then PP2 and
PP3.  The final stage sends output metadata back to PP0.  Each receiver checks
the causal sequence and the batch/token identity before running the shared
batch lifecycle.

### `StagePlacementDecision`

Required fields:

```text
request_id, source_stage, backup_stage, route_version, budget_version,
chunk_id, token_start, token_end, tier, failure_domain, utility,
copy_link, planned_commit_deadline
```

The invariant `source_stage.pp_rank == backup_stage.pp_rank` is validated at
construction and at receive time.  `tier` is `HBM`, `HOST`, or `RECOMPUTE`.
`RECOMPUTE` records an explicit uncovered chunk; it is not an absent record.

### `StageCommitAck`

Required fields:

```text
request_id, source_stage, backup_stage, route_version, budget_version,
committed_frontier_tokens, hbm_tokens, host_tokens, recompute_tokens,
commit_logical_time, failure_domain, recovery_path, copy_latency,
copy_bytes, link_id, ack_sequence
```

ACKs are monotone only within the exact tuple
`(request_id, source_stage, backup_stage, route_version, budget_version)`.
Older versions can be retained for diagnostics but cannot advance the current
safe frontier.

### `RecoveryHint`

Required fields:

```text
request_id, route_version, budget_version, failed_domain,
stage_acks[pp_rank], safe_frontier_tokens, bottleneck_pp_rank,
restored_tokens_by_stage, recomputed_tokens_by_stage,
recovery_path_by_stage, total_restore_latency, total_recompute_latency
```

For PP=4:

```text
safe_frontier_tokens = min(stage_acks[p].committed_frontier_tokens
                           for p in {0,1,2,3}
                           if version and failure-domain checks pass)
```

If any required stage is absent or invalid, its frontier is zero.  The Router
may advertise only this bottleneck prefix as recoverable.

## DP policy budget versus node physical capacity

The two layers are deliberately distinct:

1. The exact optimizer/Algorithm 3 publishes `(alpha_n, beta_n, version)` for
   each DP replica.  It decides how much protection that replica is allowed to
   request.
2. Stage-local HBM allocators enforce alpha against their own physical pools.
3. A node Host allocator receives leases tagged with `(dp_rank,
   budget_version)`.  It enforces both the DP beta limit and the node total
   capacity/bandwidth limits.
4. A policy-admissible lease can still be denied by node arbitration.  The
   denial is persisted as `unmet_tokens` and `unmet_utility`; it does not cause
   beta ownership to move to the node.
5. For every node and instant, resident Host bytes across all DPs are at most
   physical capacity.  For every DP/window, admitted Host bytes are at most its
   beta policy budget.  Both invariants are checked independently.

## Window state semantics

The following names have one meaning across HBM and Host tiers:

- `unique_offered`: unique tokens first offered in this window.
- `attempted_by_tier[t]`: tokens for which placement in tier `t` was attempted.
- `admitted[t]`: attempted tokens accepted in tier `t` during this window.
- `resident_start[t]`, `resident_end[t]`: stock at window boundaries.
- `released[t]`: resident tokens removed because their lifetime ended.
- `evicted[t]`: resident tokens displaced before normal release.
- `unmet_tokens[t]`: attempted but not admitted tokens.
- `unmet_utility[t]`: objective-weighted value of unmet tokens.
- `normal_pressure`: calibrated normal-path memory/queue pressure.
- `recovery_p99`, `slo_miss_rate`: observed outcome signals.

No field called `D`, `U`, `O`, or `Z` is used in the new schema.  Stock and
flow are never combined under one name.

## Failure and retry protocol

1. Failure injection resolves its victim set as explicit `StageId` values.
   `pp_worker` scope affects one stage; `dp_replica` affects all four stages of
   one DP; `node` affects every stage mapped to the node.
2. The recovery coordinator reads only version-matched committed ACKs that
   survive the failed domain.
3. A retry is routed once.  Each destination PP stage restores only KV produced
   by the same source PP stage and verified for the same prefix/version.
4. Missing suffixes and missing stages are explicit recomputation.  Each retry
   persists restored and recomputed tokens per stage, queue delay, transfer
   delay, recomputation delay, total recovery delay, and the bottleneck stage.
5. Workflow completion is emitted only after the normal SGLang-equivalent
   stage lifecycle completes; recovery bookkeeping cannot bypass scheduling or
   KV allocation.

## Clock isolation

The runtime state machine above is clock-agnostic.

- Online: each stage performs the existing Pipeline-v2 wall-clock advance; PP
  messages impose causal waits naturally.
- Offline: each DP replica has an independent causal event domain.  Its four PP
  stage clocks advance on compute and pipeline messages.  The parent merges
  externally visible events and failures by logical timestamp.
- There is no per-step global max synchronization across DP replicas.

## Hard validation failures

The run is invalid, not merely degraded, if any of the following occurs:

- process count or `(dp, pp, tp)` identities do not match the declared topology;
- a cross-stage Connector placement is observed;
- batch/request/token identities differ across a PP microbatch;
- a budget version is partially published across a replica;
- DP beta or node physical capacity is exceeded;
- an aggregate safe frontier is larger than any required stage frontier;
- a retry reports more restored tokens than its verified stage prefix;
- online and offline use different route plans, failure manifests, request sets,
  or placement/recovery implementations for a paired comparison.
