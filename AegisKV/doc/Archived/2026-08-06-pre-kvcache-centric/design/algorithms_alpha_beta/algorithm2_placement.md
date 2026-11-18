# Algorithm 2: topology-aware protection and recovery placement

## Decision unit and hard safety filters

The decision unit is a KV chunk.  For each request, Algorithm 2 evaluates GPU,
node-shared Host, persistent L3 and recompute candidates.  A candidate is
discarded before utility ranking when any hard condition fails:

1. the copy does not survive the declared GPU/process/node/rack failure scope;
2. capacity or a protection lease is unavailable;
3. the recovery path is not executable on the proposed retry replica;
4. the committed payload/manifest frontier is missing;
5. predicted recovery cannot satisfy the configured recovery guard.

The current runtime compatibility names are `l1_tokens = GPU protection` and
`l2_tokens = Host protection`.  New profile records use the physical names
`gpu`, `host`, `l3` and `recompute` to avoid treating cache-level names as a
topology contract.

## GPU candidate

A GPU copy is placed on a distinct DP replica.  With DP=4 and PP=4, the request
has four stage-local KV partitions.  The only directly reusable mapping is:

`source (DP d, PP p) -> target (DP d', PP p), for p in 0..3`

The ACK records all four source/target GPU pairs.  On failure, GPU tokens are
counted as restored only if retry is routed to exactly `d'`.  Routing to any
other DP rank invalidates the GPU portion; AegisKV recomputes it or uses another
durable tier.  It never models an implicit post-failure peer-to-peer transfer.

The normal-path write is `PCIe P2P` within a node or GPU/RDMA transport across
nodes.  Recovery is `direct_hbm_reuse` and therefore has no KV copy stage.

Normal execution routing remains cache/load aware.  Backup selection is a
separate decision: every healthy non-source replica is filtered by the failure
scope, then scored using the best feasible GPU/Host/L3 utility minus the
criticality-weighted retry queue delay.  The score includes target GPU
protection pressure, node Host-pool pressure, write path, recovery path, and
the exact-retry-route requirement.  Thus the backup is not simply the second
entry in the normal execution ranking.  Matched failure-manifest victims keep
their frozen execution/backup ranks so policy comparisons remain paired.

## Host candidate and Mooncake-style shared pool

Each physical node owns one process-external Host allocator, not one private
pool per GPU or scheduler process.  The SGLSim metadata model mirrors the
relevant Mooncake Store properties:

- one bounded allocator with explicit capacity and protected-use accounting;
- leases whose release returns capacity;
- aligned allocation with internal-fragmentation accounting and a bounded
  region count;
- promotion of an existing cache payload to `LOCAL_PROTECT` without copying it;
- a separate bounded pinned-memory quota;
- pageable fallback when pinned quota is exhausted;
- lower-utility lease replacement under pressure.

As in Mooncake's native buffer pool, capacity is charged to allocated rather
than logical payload size.  A 257-token lease with 256-token alignment therefore
occupies 512 tokens and reports 255 internal-fragment tokens.  Region or byte
pressure ranks leases by utility per allocated token and evicts complete
lower-density leases; if no admissible victim exists,
protection falls back to a lower tier or recomputation without blocking normal
serving.  Lease release is RAII-like in the metadata model and remains version
guarded so a stale owner cannot free a newer allocation.

The shared pool is authoritative.  When it replaces a lease, the Router
downgrades the corresponding recovery hint to recomputation before another
retry can observe it, then sends a version-guarded invalidation to the owning
scheduler.  A delayed invalidation cannot erase a newer replacement lease.

If the Host lease is on the source node, backup is `D2H`; this survives a GPU or
scheduler-process failure but not a node failure.  If the lease is on another
node, backup is `D2H + H2H` and survives a source-node failure.  At recovery:

- retry on the Host-pool node: `H2D`;
- retry on another node: `H2H + H2D`.

Unlike a GPU copy, a surviving Host lease can therefore be used by a retry
replica other than the original backup DP rank.  The Router evaluates the path
cost separately for every healthy rank.

## Recovery rehoming

A transient failure must not become a permanent load hotspot.  When a failed
rank clears volatile L1, the Router records only cache keys that previously
executed there, not keys present solely as protection copies.  Requests may
temporarily migrate while the rank is unavailable.  After recovery, the next
request for each recorded key is sent back once to rebuild its execution
prefix; subsequent requests again use normal cache/load-aware ranking.  This
one-shot rehome is recovery control, not steady-state route pinning.

## Persistent L3 and committed manifests

L3 is a cluster-external, capacity-bounded persistent protection pool.  It is
not enabled by silently turning on ordinary HiCache L3 for every policy.
Algorithm 2 sends only the residual frontier not admitted to GPU or Host:

- backup path: `D2H + L3 write`;
- recovery path: `L3 read + H2D`;
- reuse requires a live lease whose backup version, manifest version, and
  committed quorum match the Router hint;
- capacity pressure replaces only lower-utility manifests;
- eviction synchronously clears the old Router frontier before any retry can
  consume it.

The Qwen3-32B DP4/PP4 profile models a 2,000,000-token cluster-external pool
(approximately 512 GiB at 256 KiB per logical token across four 64 KiB PP
partitions) with manifest quorum 2.  This is a metadata and latency model of
the persistent tier, not a claim of measured disk I/O.

## Utility after feasibility

For a feasible candidate `c`, the current score is:

`expected_benefit(c) = Pfail * Pcritical * (overrun(recompute) - overrun(recover(c)))`

`utility(c) = expected_benefit(c) - write_cost(c) * (base_weight + pressure_weight * pressure)`

where `overrun(x) = max(x - Slack_LCB, 0)`.  Capacity pressure and path pressure
increase normal-path cost; they never turn an invalid copy into a valid one.
Recompute is always present with zero placement utility, so negative-value
backup candidates are rejected.  `Pfail` is a historical/configured failure
hazard for the protected domain; it is not derived from the future replay
manifest.  This avoids valuing every critical request as though its failure
were certain, which would over-admit slow Host/L3 writes on the normal path.

The formal DP4/PP4 profile currently uses `Pfail=0.01`, base write-cost weight
`0.35`, and pressure weight `0.50`.  These values keep fast GPU protection
eligible for high-value frontiers while requiring Host/L3 to justify their
larger write costs.

## Current implementation and evidence fields

- `python/sglsim/srt/simulation/aegis_placement.py`: topology, component path
  costs, candidate evaluation, node-shared Host leases, and committed L3
  manifests;
- `data_parallel_controller.py`: plan construction, process-external Host pool,
  per-retry-rank path evaluation and direct-route enforcement;
- `aegis_l1_l2.py`: rank-local chunk admission and authoritative ACK fields;
- WCT results: `gpu_direct_reuse_tokens`, `host_reuse_tokens`,
  `l3_reuse_tokens`, Host/L3 restore paths, restored/recomputed tokens,
  committed-pool state, and retry latency.

Standalone `l1_only`, `l2_only`, and `l3_only` profiles expose the three
physical tiers; `aegiskv` lets the utility controller combine them.
