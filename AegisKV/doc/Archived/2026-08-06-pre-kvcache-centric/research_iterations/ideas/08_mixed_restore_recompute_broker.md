# Idea 8: Mixed Restore/Recompute Recovery Broker

## Claim

At failure time, do not choose globally between “restore all” and “recompute
all.” Run restoration and recomputation concurrently over different token or
stage ranges, and join at a verified frontier chosen from current resource
conditions.

## Mechanism

Given materialized-KV frontier `K`, scope-protected/group-committed KV frontier
`C`, emitted-output frontier `E`, a scope-durable manifest, and a replacement
worker, the recovery broker builds two base-prefix pipelines:

- **recompute pipeline:** immediately rebuilds the earliest missing prefix
  using available compute;
- **restore pipeline:** fetches later committed chunks through remote Host,
  H2D, parity decode, or PP anchors.

For candidate join point `j`:

- recompute covers `[0, j)`;
- restore covers `[j, C)`;
- after their verified join reconstructs `[0, C)`, dirty-suffix replay rebuilds
  request-valid KV in `[C, K)`;
- ordinary generation resumes only after KV catches up to its required
  continuation point with compatible topology/version state.

`E` is used to suppress duplicate client output and restore sampling/streaming
state. It is not a KV-availability frontier and is not substituted for `K`.

The broker tests a bounded set of page boundaries and selects the earliest
feasible completion. It cancels late restore work when recompute overtakes it,
and cancels redundant recompute when a verified restored chunk arrives first.

## Resource model

Approximate base-prefix completion at join `j` is:

`T_base(j) = max(T_recompute(0,j), T_restore(j,C)) + T_join`.

Total recovery also includes

`T_suffix_replay(C,K) + T_output_state(E) + T_exec_ready`.

This equation is a calibration aid, not the mechanism. The implementation is a
race between chunk futures with:

- compute credits;
- remote read/RNIC/H2D credits;
- parity decode workspace;
- PP-stage dependencies;
- foreground/recovery priority.

Recovery traffic receives a different credit class from background backup,
but still preserves capacity needed to keep healthy requests alive.

## Parallelism behavior

- TP: restore/reconstruct every required logical attention shard. Replicated
  copies of one logical head are alternatives, and a coded alternative is
  usable only when its post-failure decode quorum survives.
- PP: recompute only failed stages when a compatible boundary anchor covers the
  same token extent and survives the declared stage failure; otherwise upstream
  replay expands.
- DP: route to an independently warm target only for standard full-replica DP.
  DP-attention requires topology-aware attention/EP reconstruction and is not
  automatically a spare full replica.
- EP: repair expert placement, logical weights/version, routing epoch, and
  A2A/collectives separately; attention-KV restore does not solve expert
  unavailability. If the failed rank also owns attention KV, both must recover.

## Evidence

- GhostServe explicitly combines initial recomputation with asynchronous
  parity restore/reconstruction and reports that hybrid recovery is needed in
  some regimes (physical pp. 7--8 and 11).
- HCache combines I/O and recomputation to restore exact KV from smaller
  intermediate activations (physical pp. 6--8), though its context is cache
  restoration rather than fault recovery.
- The SGLang Host/storage path has distinct D2H, storage-operation completion,
  and H2D phases, giving the broker observable futures rather than one
  indivisible copy. The proposed broker must upgrade completion to verified
  scope-specific payload quorum plus a scope-durable or rebuildable manifest
  commit.
- The profiler shows prefill compute and D2H can contend differently from
  decode, motivating runtime measurement instead of a fixed join threshold.

## Falsifiable experiment

For failures at different sequence positions, sweep:

- local versus remote Host;
- replica versus parity;
- TP/PP sizes;
- model/recompute throughput;
- copy-path congestion;
- protected frontier coverage;
- manifest/directory owner placement and rebuild latency.

Compare full recompute, full restore, static 50/50 hybrid, GhostServe-like
hybrid, and the resource-aware broker. Report failure-to-first-token,
failure-to-catch-up, healthy-request P99, canceled work, and bytes/compute
wasted by races.

The idea fails if join synchronization or duplicate work dominates, if
restored and recomputed numerical state cannot be made compatible, or if
recovery priority causes unacceptable collateral SLO damage.

## Novelty boundary

Hybrid restore/recompute is present in prior work. The potential contribution
is composing it with per-request slack, PP boundary anchors, topology-aware
scope-durable manifests, the three `K/C/E` frontiers, and the same
capacity/path-credit broker used during normal operation.
