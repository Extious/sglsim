# Algorithm 3: adaptive replica-GPU and node-Host budgets

## Control ownership

Independent `alpha` on all 16 physical GPUs is neither necessary nor safe for
DP=4/PP=4.  A directly reusable GPU backup must contain the same request on all
four target PP stages, so its usable capacity is limited by the bottleneck PP
GPU.  A target DP replica therefore owns one `alpha` controller whose budget is
applied consistently to its four stage-local GPU partitions.  The profile name
is `dp_replica_pp_bottleneck`.

Host memory is different: schedulers on one physical node share one bounded,
process-external pool.  Consequently there is exactly one authoritative
`beta` controller per node.  Rank-local managers report Host demand but cannot
publish conflicting per-GPU beta values.  The profile name is
`node_shared_host_pool`.

## Adaptive window

A fixed request count is unstable under heterogeneous Real CC workflows.  Both
controllers now use a bounded multi-trigger window:

1. never update before `window_min_requests`;
2. update at `window_target_ms` after the minimum sample count;
3. update early on an overflow shock;
4. always update by `window_max_requests`.

The formal DP4/PP4 profile uses min=4, max=16, target=5 s, and a 256-token
overflow trigger.  Every update records the trigger and actual elapsed time.

## Signals and next-window update

For each attempted tier, Algorithm 2 emits request-local:

- `D`: tokens for which that tier was actually attempted;
- `U`: tokens authoritatively admitted;
- `O`: attempted tokens not admitted.

Alternative tiers that were never attempted are not double-counted as demand.
Replica alpha consumes rank-local L1 `D/U/O`, normal serving pressure, and an
occupancy floor.  Node beta consumes post-shared-pool Host `D/U/O`, the maximum
observed normal Host pressure, and the current protected-use floor.

Node beta predicts a token rate with EWMA and converts it to next-window demand:

`predicted_tokens(K+1) = EWMA(required_tokens / elapsed_time) * target_window_time`

For Host memory, `D/U/O` use allocator occupancy rather than only useful
payload tokens.  Requests are rounded to `host_alignment_tokens`; the ACK also
retains logical admitted tokens and internal-fragment tokens.  Thus beta sees
the same physical capacity units enforced by the shared pool and cannot hide
fragmentation behind a smaller logical-token count.

The published ratio is smoothed, hysteresis-filtered, and bounded by the normal
serving safety cap.  It may fall below current protected occupancy: before the
new ratio becomes externally usable, the controller revokes the lowest-utility
leases until occupancy satisfies the new budget.  Router hints are downgraded
synchronously and target schedulers receive version-guarded invalidation
commands.  Replica alpha uses the same value-ordered revocation rule for L1.

## Safety invariants

- alpha cannot reserve capacity needed by normal L1 serving;
- beta cannot exceed the node-shared Host safety cap;
- no Host lease can remain reusable after shared-pool eviction;
- a delayed invalidation cannot clear a newer lease version;
- control updates change protection admission only, never SGLang batch
  scheduling or KV correctness.
