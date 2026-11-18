# Idea 3: Failure-Domain Adaptive KV Coding

## Claim

Choose local Host, remote Host replication, or erasure-coded parity according
to the requested failure scope and current topology. “Host backup” without a
fault-domain contract is only a location, not a protection guarantee.

## Action set

| Action | Extra payload | Typical coverage | Restore path |
| --- | ---: | --- | --- |
| Recompute | 0 | none | model replay |
| Local Host lease | one local copy or zero-copy promotion | GPU/device; process only with an external owner | local H2D |
| Remote replica | one full remote copy | source-node | RNIC + H2D |
| `(k,m)` parity | `m/k` aggregate parity overhead | declared failure only if at least `k` valid fragments survive its fault-domain loss | gather surviving shards + decode + H2D |
| Peer HBM emergency | one peer copy | narrow soft-GPU fault only | fast P2P |

Algorithm 2 first filters actions by failure scope, then by deadline and
resource feasibility. It never treats same-node peer HBM as node-failure
protection.

## Stripe construction

The coding unit is not an arbitrary set of pages. A stripe contains:

`<request, topology_epoch, token_extent, unique logical attention-TP/PP
component IDs, encoding_version, k, m, fault-domain map, fragment checksums,
commit generation>`.

Only unique logical KV shards enter the stripe. Replicated KV heads created
when effective attention TP exceeds the number of KV heads are alternative
copies of the same logical component and are deduplicated before protection.
Each fragment carries enough of this header to rebuild the stripe directory
after loss of a controller or worker process.

Parity placement obeys:

1. after any declared failure-domain loss, at least `k` checksum-valid fragments
   and a scope-durable stripe manifest must survive;
2. a request frontier advances only when the data/fragment ACK set proves that
   surviving decode quorum, not merely that one parity write completed;
3. topology change seals the old stripe and starts a new epoch;
4. partial stripes are either padded/versioned or remain recompute-only;
5. recovery checks that enough checksum-valid fragments and the stripe manifest
   survive the declared fault-domain loss before decoding.

A single remote parity shard does **not** protect source-node loss when all
`k` data shards are on that source node. Node scope requires either a full
remote replica or a coded placement whose fragment distribution still leaves a
decode quorum after the entire node disappears. Rack scope applies the same
test after removing the entire source rack.

## Adaptive decision without a large formula

Use a small state machine:

- `GPU_ONLY` scope and ultra-low slack: local Host first; peer HBM only if
  local Host cannot meet the deadline and spare HBM/path credits are measured.
  Worker-process scope additionally requires an external Host owner or another
  process/domain.
- `NODE` scope: a full replica outside the source node for small/urgent
  extents; coding for large, stable extents only when at least `k` fragments
  remain after source-node loss and encoding/gather deadlines fit.
- `RACK` scope: place a full replica outside the source rack, or distribute
  coded fragments so at least `k` remain after source-rack loss.
- high churn or tiny attention-TP group: prefer replication/recompute because
  stripe completion and encoding overhead dominate.
- high attention TP with large stable chunks: parity becomes attractive.

## Evidence

- GhostServe protects TP KV shards with Host-memory parity and explicitly
  avoids persistent peer-HBM storage; it reports parity benefits at larger TP,
  while its benefit can disappear at TP=2 and hybrid recovery is sometimes
  needed (physical pp. 5--8 and 11).
- The live SGLang remote backend sends Host pages to one centralized DRAM
  server but does not replicate or code across servers. One ACK is therefore
  one failure domain, not rack-level durability; see
  [live storage audit](evidence/live_sglang_storage_audit.md).
- Mooncake and Strata show that remote memory/I/O placement and topology are
  performance resources, but neither provides a fault-tolerance contract; see
  [KV review](../../related_works/kvcache_backup/per-paper-review.md).
- AdaCheck and other training work demonstrate that redundancy already present
  in parallel execution can reduce checkpoint payload, but the exact
  recoverability contract must remain explicit.

## Theoretical cost boundary

For a full replica of payload `B`, storage overhead is `B` and transfer is
approximately `B / bw`.

For a systematic `(k,m)` code, aggregate parity storage is `(m/k)B`, but the
critical path also contains encoding and potentially multi-source gather:

`T_code = T_encode + max_path(T_parity_write) + T_quorum_ACK`.

Coding is selected only when `T_code` fits the protection deadline and the
declared fault-domain loss leaves both a decode quorum and recoverable manifest.
Storage savings alone are insufficient.

## Falsifiable experiment

Sweep effective attention TP `T_a` in `{1,2,4,8,16}`, extent size,
source/rack topology, and failures:

- GPU reset;
- worker/process loss with Scheduler-owned versus external Host ownership;
- manifest/directory owner loss;
- node power loss;
- parity-server loss;
- correlated rack loss.

Compare recompute, local Host, remote replica, static parity, and adaptive
coding. Measure useful committed bytes, no-failure interference, recovery
tail, encoding/gather traffic, surviving data/manifest quorum, directory rebuild
time, and false protection claims.

The idea fails if parity completion tails dominate realistic slack, if small
stripes waste too much padding, or if correlated failure-domain placement
cannot be enforced by the runtime.

## Novelty boundary

Erasure coding itself is prior art. The proposed novelty is the
deadline-constrained, topology-epoch-aware selection and placement of
replication/coding inside a shared KV Host pool, with explicit downgrade to
recompute and a scope-durable, rebuildable coding manifest.
