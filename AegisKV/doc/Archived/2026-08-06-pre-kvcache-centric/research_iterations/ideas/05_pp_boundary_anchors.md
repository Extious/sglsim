# Idea 5: Pipeline-Parallel Boundary Anchors

## Claim

For pipeline-parallel serving, protect a compact activation anchor at selected
PP boundaries so a failed stage can rebuild only its own KV instead of
replaying all upstream stages.

## Mechanism

For each protected token extent and selected producer boundary `s`, store:

`<request, topology_epoch, exact token_extent, producer_boundary_id,
consumer_stage_id, input activation tensor for every token in the extent,
position/attention metadata, activation format, checksum, storage-owner fault
domain, scope-valid manifest/ACK evidence>`.

Stage-local recovery is valid only when the anchor covers exactly the token
extent whose consumer-stage KV is regenerated and remains readable after the
declared consumer-stage failure. A pointer or descriptor that dies with the
stage is not an anchor commit.

After the consumer stage fails:

1. replace/restart the stage and restore its model shard;
2. fetch the latest topology-compatible anchor covering the same token extent
   and surviving the failed stage's fault domain;
3. replay only the failed consumer stage's layers over exactly that extent;
4. regenerate its K/V pages;
5. join the request at the minimum compatible frontier across all required
   PP stages and logical attention shards;
6. use ordinary upstream replay for any suffix beyond the anchor.

Anchor placement is another Algorithm 2 action. It is selected only when the
saved upstream replay exceeds the anchor's capacity and transfer cost.

## Size argument

For one bf16 hidden vector of dimension `d`, a boundary activation costs

`B_anchor/token = 2d` bytes.

For a PP stage with `L_s` attention layers, `H_kv` KV heads and head dimension
`D`, its bf16 KV costs

`B_stage_KV/token = 2 * L_s * H_kv * D * 2` bytes

before TP sharding.

Concrete Qwen3-8B example:

- hidden size `d=4096`: anchor = 8 KiB/token;
- 36 layers split over PP=4 gives 9 layers/stage;
- 8 KV heads, head dimension 128: stage KV = 36 KiB/token;
- one raw boundary anchor is therefore about 22% of one stage's KV.

This is not automatically a 4.5x system saving: metadata, padding, TP
distribution, activation precision, and replay compute matter. It establishes
only that a boundary anchor can be materially smaller than duplicating a
stage's full KV.

## Placement policy

- protect boundaries before stages with high replay cost or high failure rate;
- coarsen anchors for long slack and cheap upstream compute;
- choose local Host for GPU/device-state failure; use it for process failure
  only with a process-external owner and durable/rebuildable directory; for node
  scope use storage outside the node or a coded placement whose decode quorum
  survives node loss;
- deduplicate an anchor shared by multiple agent descendants;
- expire it when all dependent requests pass a newer anchor or terminate.

The controller can cap anchor capacity inside the same unified Host pool rather
than creating a new memory tier.

Anchor descriptors follow the same scope-durable manifest rule as KV extents:
the compact commit record must survive the declared fault, or the directory
must be deterministically rebuildable from self-describing surviving anchors.
A Scheduler-local-only anchor directory does not prove recoverability.

## Evidence

- HCache stores intermediate hidden activations and reconstructs exact KV with
  combined I/O and GPU computation, demonstrating the core
  activation-to-KV transform (physical pp. 1, 6, 8). It is a restoration/cache
  paper, not fault-tolerance evidence.
- MoEvement logs activations/gradients at PP boundaries in its training
  recovery design, showing that stage-boundary state can decouple recovery
  across pipeline segments. Training semantics are not directly reused.
- The live SGLang audit shows PP divides KV by layers but request recoverability
  still needs a coherent same-token-extent frontier across all required stages
  and logical attention shards.
- The profiler shows prefill/replay is compute-dense on the tested model, so
  avoiding upstream replay can be valuable; it does not prove the same ratio
  under PP.

## Falsifiable experiment

Run PP in `{2,4,8}` with controlled stage failures. Compare:

- full-request replay;
- full failed-stage KV replica;
- every-stage boundary anchor;
- adaptive sparse boundary anchors;
- hybrid anchor plus suffix recompute.

Report Host/remote bytes, recovery latency, recomputed layer-tokens, no-failure
transfer overhead, and global frontier loss. Sweep hidden size, GQA ratio,
sequence length, link topology, anchor-owner process loss, and source-node loss.

The idea fails if transferring anchors contends more than replay, if stage
restart/model loading dominates recovery, or if exact activation state is too
large/complex after quantization and fused kernels.

## Novelty boundary

Activation checkpointing and HCache-style restoration are prior work. The
potential novelty is selecting and protecting serving PP boundary anchors by
request slack, exact token extent, and declared failure domain, then composing
only scope-surviving anchors with logical-shard KV frontiers and stage-local
recovery.
