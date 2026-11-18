# Pinned SGLang DP x PP runtime mapping

Status: implementation contract, audited 2026-08-04.

This document records the live source anchors that constrain the AegisKV
DP x PP refactor.  It is not a design inferred from diagrams.  The simulator
must mirror the control flow below without importing `sglang` at runtime; the
only intentional replacement remains CUDA/Triton/NCCL model forward with the
SGLSim latency predictor.

## Audited revisions and baseline

- SGLSim: branch `aegiskv`, commit
  `ca1fe76e88959a2bac0f6060e3042dc2c7d4f855`.
- Pinned SGLang: branch `profiler`, commit
  `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1`.
- Both trees were already dirty.  The SGLSim audit observed 485 status entries;
  the pinned SGLang tree had a local Mooncake Store change and an untracked
  `benchmark/aegiskv_mooncake/` tree.  Those edits are user-owned and are not
  part of this refactor.
- Pre-refactor focused baseline:
  `test_data_parallel_controller.py`, `test_offline_cluster.py`,
  `test_offline_coordinator.py`, `test_aegis_l1_l2.py`, and
  `test_aegis_placement.py`: **63 passed**.

Line numbers below are audit anchors for the stated live revisions; function
names and behavioral tests are the durable references.

## Source-to-simulator map

| Concern | Pinned SGLang source and function | Required behavior | Current SGLSim mapping and gap |
|---|---|---|---|
| Engine launch | `srt/entrypoints/engine.py::_launch_scheduler_processes` (around line 592), `_calculate_rank_ranges` (1401) | Without DP, launch one Scheduler process for every local PP x TP rank.  With DP, delegate worker launch to the DP controller. | `srt/entrypoints/engine.py::launch_subprocesses` builds only DP endpoints and receives a flat one-process-per-DP list. |
| DP-owned worker group | `srt/managers/data_parallel_controller.py::launch_tensor_parallel_group` (592) | For each DP replica, compute local PP and TP ranges and launch `run_scheduler_process(..., pp_rank, dp_rank, ...)` for every PP x TP rank.  DP=4, PP=4, TP=1 therefore means 16 Scheduler processes. | `srt/managers/data_parallel_controller.py::launch_scheduler_processes` (2443) loops only over `dp_rank`; `launch_scheduler_process` starts one child. |
| Scheduler identity | `srt/managers/scheduler.py::run_scheduler_process` (4574), `configure_scheduler_process`, `dispatch_event_loop` (4478) | Every Scheduler has explicit DP/PP/TP identity.  `pp_size > 1` selects the PP event loop. | `srt/managers/scheduler.py::run_scheduler_process` (2899) and `Scheduler.__init__` accept only `dp_rank`; `pp_size` changes timing mode but not process topology. |
| PP batch lifecycle | `srt/managers/scheduler_pp_mixin.py::event_loop_pp` (68), `init_pp_loop_state` (558), `_pp_launch_batch` (1265) | Stages execute in the same order.  Each microbatch receives requests and proxy state from the previous stage, runs the shared batch lifecycle, processes the prior result, and forwards request/proxy/output state around the PP ring.  `running_batch`, `last_batch`, and microbatch slots stay causally aligned. | There is no SGLSim PP mixin or PP ring.  Merely spawning four independent schedulers would duplicate admission and diverge batch/KV state, so it is forbidden. |
| PP metadata transport | `scheduler_pp_mixin.py::_pp_send_pyobj_to_next_stage` (952), `_pp_recv_pyobj_from_prev_stage` (968), `_pp_send_dict_to_next_stage` (1016), `_pp_recv_typed_dict` (1041) | Request metadata and proxy/output metadata have separate, ordered stage-to-stage channels.  The last stage returns output state to stage 0. | `srt/utils/network.py::build_dp_ipc_registry` (45) registers only DP ingress and DP detokenizer reply queues.  Stage ring and control endpoints are absent. |
| Rank groups | `srt/distributed/bootstrap.py::_init_parallel_groups` (211), `srt/distributed/parallel_state.py::initialize_model_parallel` (2105) | Within a DP replica, global worker rank is formed from PP and TP rank; PP groups connect matching TP ranks across stages. | `srt/distributed/parallel_state.py::Topology` describes DP x PP x TP metadata but does not create runtime groups or transport. |
| Layer/KV ownership | `srt/model_executor/model_runner_components/layer_setup.py::resolve_layer_indices` (113), `_resolve_pp_layer_range` (182); `srt/mem_cache/memory_pool.py` | A PP stage owns its model layer interval and indexes only that interval's KV buffers.  KV capacity and placement accounting are therefore stage-local. | The current single Scheduler owns one whole-model metadata KV pool.  A real PP refactor needs one stage-local pool per `(dp_rank, pp_rank)`. |
| Same-stage Connector | `srt/disaggregation/common/conn.py` around lines 540-564 and 1197-1280 | When source and destination PP sizes match, `target_pp_ranks = [self.pp_rank]`; bootstrap lookup includes the target PP rank. | `AegisTopology.pp_gpu_mapping` encodes a logical same-stage mapping, but `AegisBackupPlan` and placement are request-level and do not execute per stage. |
| All-stage transfer completion | `srt/disaggregation/nixl/conn.py::TransferStatus` (309), notification parsing around 2169 | Track received and expected chunks by `pp_rank`; a transfer is complete only when every expected stage has completed. | `AegisBackupAck` has one aggregate frontier.  It can currently claim recoverability without proving every required PP stage committed the prefix. |

## Non-negotiable runtime invariants

1. `scheduler_process_count == dp_size * pp_size * tp_size` for the supported
   local topology.  DP=4, PP=4, TP=1 means exactly 16 independently alive
   Scheduler/worker processes.
2. Public request ingress is to PP0 of the chosen DP only.  Later stages learn
   the request and immutable route through the ordered PP channel; they do not
   independently route or admit an external copy.
3. For every microbatch, all stages agree on batch id, request ids, forward
   mode, token frontier, and route version.  A mismatch is a hard error with a
   persisted diagnostic event.
4. Every stage owns only its layer interval and stage-local KV metadata.  No
   `(source_dp, pp_i)` KV chunk may be stored as `(backup_dp, pp_j)` for
   `i != j`.
5. A recoverable prefix is the minimum committed, version-matched frontier
   over all required PP stages.  Aggregate counters may never exceed this
   bottleneck frontier.
6. Online and offline modes share request scheduling, placement, ACK,
   failure/recovery, and retry code.  They differ only in the clock/driver:
   real wall-clock sleeping versus causal logical-event advancement.
7. Logical clocks are independent across DP replicas.  PP stages within one
   replica synchronize only through causal pipeline messages; there is no
   strict max-clock barrier across DPs.
8. Shutdown, crash detection, restart, and failure injection operate on an
   explicit `(dp_rank, pp_rank)` stage group.  A DP- or node-scope failure is
   represented as a set of stage failures, not an implicit scalar rank.

## Required verification before experiments

- Unit tests for rank enumeration, stage endpoints, immutable message schemas,
  stage ownership, prefix aggregation, and node-capacity arbitration.
- Tiny deterministic PP lifecycle replay that checks request/batch/token order
  at every stage.
- Integration launch/clean shutdown for DP=2, PP=2 and DP=4, PP=4, including a
  single-stage crash and a whole-replica crash.
- Online/offline parity gates on identical trace, seed, route plan, and failure
  manifest.  Exact timestamp equality is not required; route, completion order,
  event counts, forward/communication accounting, and distributional metrics
  must satisfy the registered tolerances.

No WCT result produced by the old one-Scheduler-per-DP PP=4 topology is formal
evidence for the new DP x PP system.  It remains useful only as smoke or
calibration history.

## Implementation checkpoint (2026-08-04)

- `srt/managers/pipeline_parallel.py` now defines explicit stage identities,
  immutable route/budget/batch/placement/ACK/recovery contracts, all-stage
  frontier aggregation, and a validated parent process group.
- `launch_pipeline_scheduler_processes` launches and cleans up one child per
  DP x PP x TP identity.  A real DP=2, PP=2 smoke observed four ready children
  and four zero exit codes; the lifecycle is retained as a unit test.
- `Scheduler` now has explicit `dp_rank`, `pp_rank`, and `tp_rank`, binds a
  stage-specific ingress, and restricts client-visible output to PP0.
- The default Engine still uses the pre-existing one-Scheduler-per-DP launcher.
  This is intentional until the request/proxy/output ring and batch lifecycle
  checks are integrated.  The process-count smoke is not a PP-serving claim.
- Current runtime regression: `test/unit/srt`: 413 passed.  Top-level
  `test/unit` collection is independently blocked by seven legacy benchmark
  tests importing the already-absent `AegisKV.draft` package.
