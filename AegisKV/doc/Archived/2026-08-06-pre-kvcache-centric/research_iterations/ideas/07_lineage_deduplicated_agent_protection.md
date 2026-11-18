# Idea 7: Lineage-Deduplicated Multi-Agent KV Protection

## Claim

Represent shared prompts, tool transcripts, and parent-agent prefixes as
content-addressed immutable KV objects. Protect one object per lineage and use
dependency refcounts instead of separately backing up every descendant
request.

## Mechanism

Split session state into:

- immutable shared prefix objects;
- branch-private append-only suffix extents;
- mutable scheduler/sampling state in the recovery manifest.

Each object has:

`<content hash, model/layout version, token range, parent object, child
refcount, protection lease, fault-domain ACKs>`.

On branch creation:

1. child requests reference the parent's committed prefix object;
2. no payload copy occurs;
3. only the private suffix starts a new dirty frontier;
4. the parent protection lease persists until every dependent child passes a
   newer recovery point or terminates;
5. garbage collection is driven by the workflow DAG, not only request
   completion time.

## Resource and correctness analysis

If `m` children share prefix size `P` and have suffixes `S_j`, naive
per-request protection stores

`mP + sum(S_j)`.

Lineage dedup stores

`P + sum(S_j)`

before redundancy coding. The maximum raw saving is `(m-1)P`, but only if KV
layout, model version, positional semantics, and content hash are identical.

Dedup must not cross:

- different model/adapter versions;
- incompatible KV layouts or quantization;
- differing position IDs or context-dependent transformations;
- security/tenant boundaries without explicit authorization.

An object is recoverable only after its own protection ACK; a child reference
does not make an uncommitted parent durable.

## Why it is specifically multi-agent

Agent systems generate structured sharing:

- one system prompt feeds many specialists;
- a planner's transcript becomes the prefix of several tool agents;
- retry/speculative branches share all state up to a decision point;
- finished parents may remain necessary long after their HTTP request ends.

This makes DAG lifetime and fan-out a lower-level cache/protection signal,
while Algorithm 1 still supplies criticality and slack.

## Evidence

- CacheBlend, Cache-Craft, Mooncake, Marconi, SYMPHONY, and production KV-cache
  characterization demonstrate reusable prefixes/chunks and the importance of
  admission, layout, and lifecycle; none should be called fault tolerance.
  See the [KV review](../../related_works/kvcache_backup/per-paper-review.md).
- The unified Host pool makes a cached content object promotable to a
  protection lease without a duplicate D2H copy.
- The ACKed-prefix invariant and topology/version manifest prevent a
  content-address hit from being mistaken for a recoverable checkpoint.

## Falsifiable experiment

Construct agent DAGs with controlled fan-out, shared-prefix fraction, branch
lifetimes, and tenant boundaries. Compare:

- per-request protection;
- radix-prefix page sharing only;
- content-addressed sharing without DAG lifetime;
- lineage-aware protection/refcounting.

Measure physical bytes, D2H/RNIC bytes, lookup/hash CPU cost, protection hit
rate, stale/collision rejection, parent retention tail, and recovery WCT.

The idea fails if context-dependent KV invalidates most apparent sharing, hash
and metadata cost dominates, or long-lived descendants pin excessive Host
capacity.

## Novelty boundary

Prefix caching and content addressing are prior work. The potential novelty is
making an agent DAG's dependency lifetime control **fault-tolerance leases and
remote ACK retention**, while protecting branch-private suffixes with separate
slack-bounded frontiers.
