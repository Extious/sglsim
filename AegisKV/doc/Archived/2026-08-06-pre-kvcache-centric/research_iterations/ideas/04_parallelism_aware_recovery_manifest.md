# Idea 4: Parallelism-Aware Minimum-State Recovery Manifest

## Claim

KV pages alone are not a recoverable distributed request. Publish a compact,
topology-epoch-specific manifest that defines the minimum coherent state under
TP, PP, DP, and EP.

## Required manifest

At minimum:

`<request/session ID, request generation, topology epoch, model/adapter version,
materialized-KV frontier, scope-committed KV frontier, emitted/output-token
frontier, sampling/RNG and streaming-offset state, scheduler sequence state,
logical attention-TP shard/replica map, PP stage map, DP owner/eligible reroute
targets, EP logical-expert/weight version and placement/routing epoch,
page/fragment checksums, coding/decode quorum, pool-owner epoch, fault domains,
protection ACKs, commit-record locations>`.

The three frontiers are different:

- `K_i`, materialized KV available as a protection source;
- `C_i`, the longest scope-durable, group-committed KV prefix;
- `E_i`, output already exposed to the client.

The dirty protection suffix is `K_i - C_i`, not `E_i - C_i`. A target must not
re-emit `E_i`, and it must not assume every emitted token has a materialized or
protected KV page. Physical KV slot indices are local allocation details;
failover reallocates them and rebuilds request-to-token-to-KV maps.

The request-level protected frontier is:

`C_request = min_{required PP stage, logical shard}(C_component_scope_safe)`

over every **logical** component required by the active topology epoch. For a
logical shard with replicated copies, those copies are alternatives: its
scope-safe component frontier is the longest prefix that still has a surviving
copy or decodable fragment set after the declared fault. The request frontier
then takes the minimum across required logical shards and PP stages. A
rank-local ACK cannot advance it.

## Scope-durable manifest and directory

The manifest/directory is itself recovery state. It must not exist only in the
failed scheduler, tokenizer manager, DP controller, or source node.

Commit therefore requires both:

1. a data quorum that remains decodable after the declared fault-domain loss;
2. a compact manifest/commit record replicated or coded at the same or stronger
   failure scope.

Each page, coding fragment, or PP anchor also carries a self-describing header
with request generation, token extent, logical component, topology/model epoch,
checksum, coding set, and fault domain. A replacement controller can rebuild
the directory from surviving headers and commit records. Co-storing a manifest
only with data in the same failed domain is not durable; a centralized
directory without a reconstruction path is a single point of failure.

## TP

- When effective attention TP `T_a` divides `H_kv` and `T_a <= H_kv`, each rank
  owns a smaller shard and aggregate physical KV equals the unique logical KV.
- When `T_a` exceeds KV heads, SGLang gives each GPU at least one head and
  replicates KV heads. Protect unique logical heads once; otherwise the
  protection footprint grows unnecessarily.
- One rank failure stalls collectives. KV recovery must be paired with worker
  replacement, weights, communicator reconstruction, and scheduler remapping.
- Parity is useful for unique shards at sufficiently large TP, but not a
  substitute for control-plane reconfiguration.

For a standard MHA/GQA model, the live-code aggregate physical size is:

`B_phys/token = L * (D_k + D_v) * element_size *
T_a * max(1, floor(H_kv/T_a))`.

For bf16 with `D_k = D_v = D`, this is
`2 * L * D * 2 bytes * T_a * max(1, floor(H_kv/T_a))`. These formulas assume a
supported divisible configuration when `T_a <= H_kv`; `T_a` is not global TP
or EP size.

The audited Qwen3 configs give 114,688 B/token for 0.6B, 147,456 B/token for
8B, and 262,144 B/token for 32B at unique logical size; aggregate physical
amplification begins when `T_a > H_kv`.

## PP

- Each stage owns KV for its layer interval; aggregate KV remains unchanged.
- A request checkpoint is useful only if every needed stage has a compatible
  token frontier.
- Recovering only the failed stage requires either its KV or an upstream stage
  boundary anchor covering the same token extent, stored outside the failed
  stage's declared fault domain, from which its KV can be regenerated.
- Without the anchor, recovery may replay from the model input or a much
  earlier stage despite other stages' intact KV.

## DP

- Standard full-replica DP supplies spare compute/reroute targets, not shared
  request state by itself. An eligible target must survive the declared fault
  domain and reserve capacity for the failed owner's load.
- SGLang `enable_dp_attention` is not an independent full replica: it
  repartitions the global TP group into nested attention DP/CP/TP and MoE
  DP/EP/TP groups. Its recovery must use those group manifests rather than
  assuming a hot full-model DP copy.
- Copying all KV into a live replica's HBM consumes the same resource needed
  to absorb failure-induced load.
- Prefer a shared remote Host object plus fast reroute; content-address shared
  prefixes across DP replicas and retain only per-session suffix state.
- The manifest records the current owner and eligible reroute group.

## EP

- Attention KV follows attention-TP/PP placement, not expert ownership.
- An expert failure needs the logical expert/weight version, placement/routing
  epoch, expert shadow or reload path, and A2A/collective repair. Extra KV
  copies do not repair that failure.
- If a physical EP rank also owns an attention shard, recovery must restore both
  its logical attention-KV component and expert executability.
- A placement change may be remapped when logical experts and weights are
  equivalent; an incompatible expert/weight epoch is rejected. KV is not
  replicated once per expert.

## Evidence

- The live SGLang path already all-reduces backup completion with `MIN` across
  physical TP ranks and replicates KV heads when attention TP exceeds KV heads;
  that minimum is safe but can be unnecessarily conservative for equivalent
  logical-head replicas. See
  [live storage audit](evidence/live_sglang_storage_audit.md).
- GhostServe's direct design operates on TP data shards and shows that parity
  value depends on TP size (physical pp. 5--8 and 11).
- Context Parallelism and disaggregated-KV papers in the
  [KV review](../../related_works/kvcache_backup/per-paper-review.md) show that
  distributed execution/sharding is not a redundant checkpoint.
- MoEvement's training design separates sparse expert state and pipeline
  boundary logging, supporting the principle that different parallel
  dimensions require different recovery state.

## Falsifiable experiment

Inject one rank/stage/expert/DP-owner failure into TP, PP, standard DP,
DP-attention, EP, and hybrid configurations. Separately kill the scheduler,
manifest-directory owner, and source node.

For each, compare:

1. raw KV pages only;
2. KV plus manifest;
3. manifest plus PP boundary anchors;
4. full process checkpoint.

Report recovery success, stale-topology rejection, bytes protected, restart
time, communicator/routing time, directory rebuild time, surviving
data/manifest quorum, and replayed tokens/layers.

The idea fails if the manifest cannot be reconstructed after its declared
failure without synchronizing the request critical path, if topology epochs
change faster than protection can commit, if a data quorum survives without
the metadata needed to interpret it, or if required control state approaches
full process-checkpoint size.

## Novelty boundary

Distributed checkpoint manifests are not new. The potential contribution is a
minimal serving-specific manifest that composes KV extents, token frontiers,
logical TP/PP/DP/EP topology epochs, and a scope-durable rebuild contract while
allowing data protection and worker replacement to remain separate subsystems.
