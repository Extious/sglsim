# Offline GlobalDES failure-domain contract

Status: implementation/audit note, not paper performance evidence.  The
P1--P12 frozen paper matrix continues to use only `worker_*` and `node_*`
scenarios.  This note defines why `gpu_*` and `process_*` must not silently be
treated as worker aliases.

## Topology authority

Pinned SGLang launches one Scheduler subprocess for every local
`(dp_rank, pp_rank, tp_rank)` tuple, with one physical GPU selected for that
tuple:

- `/home/comp/zhanzhao/sglang/python/sglang/srt/managers/data_parallel_controller.py`,
  `DataParallelController.launch_tensor_parallel_group`;
- `/home/comp/zhanzhao/sglang/python/sglang/srt/entrypoints/engine.py`,
  process launch loop; and
- `/home/comp/zhanzhao/sglang/python/sglang/srt/managers/scheduler.py`,
  `run_scheduler_process`.

The current offline SGLSim adapter deliberately runs one metadata Scheduler
process per `(dp_rank, pp_rank)` and projects its symmetric TP view onto the
complete physical TP GPU group.  Thus an offline process is a `(dp, pp)`
process plus its TP group, while the physical resource model still contains
every `(dp, pp, tp)` GPU separately.

## Exact semantics

| Scope | Failed Scheduler service | HBM physically lost | Node DRAM lost | Recovery-source rule |
|---|---|---|---|---|
| `worker` | every PP process of one DP replica | every PP x TP GPU of that DP | no | all failed groups unavailable |
| `process` | one `(dp, pp)` offline process | its complete TP GPU group | no | that full TP group unavailable |
| `gpu` | the owning `(dp, pp)` process, hence the DP service is unavailable | exactly one `(dp, pp, tp)` GPU | no | the now-incomplete TP group is forbidden as a recovery source; surviving shards are not a complete copy |
| `node` | every process whose physical group is affected | every GPU physically on the node | exactly that node pool | every touched complete or partial group is unavailable |

All target resolution happens before event injection and records concrete
physical GPU keys, Scheduler process IDs, affected DP ranks, and node DRAM IDs.
Repair reloads only model/runtime baselines for targeted GPUs and an empty
volatile DRAM pool for a targeted node.  It never resurrects KV marked
`LOST`/`ABORTED`.

At equal timestamps, failure injection uses `EventPriority.FAILURE == 0`,
ahead of transfer start/completion, compute completion, and request arrival;
the monotonically increasing sequence is the final deterministic tie-breaker.

## Implemented boundaries

- `offline_failure_domain.py` resolves all four scopes into exact physical
  IDs without mutating runtime state.  Ambiguous missing `pp_rank`/`tp_rank`
  selectors fail closed.
- `OfflineResourceBridge` has exact physical lifecycle operations for a
  worker, one process group, one GPU, and one node.  A single-GPU loss clears
  ordinary L1 for the incomplete TP group but preserves independent Aegis
  copies on surviving healthy GPUs; the incomplete group remains unusable.
- The v3 runtime compiler now embeds `resolved_physical_target` in every hard
  failure event and install payload.
- The current `OfflineSchedulerCluster` injection path still fences/restarts a
  whole DP rank.  Therefore `process` with PP > 1 and `gpu` with PP > 1 or TP >
  1 are rejected by readiness.  They are accepted only in the degenerate
  PP=TP=1 topology where their exact target is identical to the worker target.
- A node target is exact for the dense formal topology, where a complete DP
  replica is node-local.  A cross-node TP group is resolved but rejected by
  the current symmetric-snapshot bridge before mutation.

## Minimal cluster integration still required

To make non-degenerate `gpu` and `process` scopes executable, the cluster API
must accept `pp_rank` and `tp_rank`, call the resolver before scheduling, and
persist its resolved target in `failures.jsonl`.  Injection must then:

1. fence the entire affected DP service and collect all active request IDs;
2. terminate only the resolved offline `(dp, pp)` process(es);
3. clear stale PP envelopes/request ownership without widening physical HBM
   loss;
4. call `fail_process_group` or `fail_gpu` on the resource bridge;
5. invalidate any PP x TP commit/recovery group missing one shard;
6. restart only terminated process(es), with empty Scheduler KV state; and
7. keep the DP route fenced until every required process and physical GPU is
   healthy and the empty-state restart ACK is observed.

Until those seven steps are integrated and tested, configuration must not
claim that non-degenerate `gpu`/`process` experiments are executable.
