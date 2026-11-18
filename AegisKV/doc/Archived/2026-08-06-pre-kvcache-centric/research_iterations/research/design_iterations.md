# AegisKV system modeling iterations

> **Iteration record; earlier rounds are intentionally preserved.** The later
> audit found additional fundamental issues: Slack is a failure-time recovery
> budget rather than a copy deadline, the default physical substrate should be
> a typed unified Host pool rather than persistent peer HBM, local Host
> ownership determines process-fault coverage, and the manifest is itself
> recovery state. The authoritative post-audit model is
> [`final_modeling_report.md`](final_modeling_report.md). Do not use the older
> `alpha=L1 / beta=L2`, “local Host covers process,” produced-frontier, or
> single-parity statements below as current semantics.

## Scope and non-negotiable invariants

- Algorithm 1 remains the application-facing step: it emits a request-level protection ticket containing conservative recovery slack, criticality, KV extent/version, and failure scope. It does not choose a backup tier or transfer route.
- Algorithm 2 is the online system mechanism. It must choose `L1 Backup`, `L2 Backup`, or `Recompute`, and for a backup it must also choose a feasible route and execution window.
- Algorithm 3 is the per-node, slow-timescale resource governor. In window `K` it consumes sealed telemetry and publishes the versioned `alpha/beta` budgets for `K+1`.
- `alpha_n` keeps its existing meaning: the fraction of manageable L1 HBM reserved for L1 Backup on node `n`.
- `beta_n` keeps its existing meaning: the fraction of manageable L2 Host DRAM reserved for L2 Backup on node `n`.
- A backup is usable only after version, checksum, fault-domain, and contiguous-prefix ACK validation. All other states fall back to recomputation. Resource control affects performance, not correctness.
- Normal inference traffic can revoke backup bandwidth credits. Backup must never require an unbounded synchronous action on the request critical path.

## Round 1 — Application-level score plus L1/L2 choice

**Model.** Rank KV blocks by a formula combining criticality, expected recovery saving, and copy cost; place the best blocks in L1 or L2.

**What it fixed.** It linked Algorithm 1 slack to Algorithm 2 placement.

**Why it was rejected as the final model.** It treated storage capacity as the only scarce resource. It could declare an L1 or L2 placement feasible even when the D2H copy engine, PCIe path, NIC, or destination memory bus was saturated. It looked like an application policy rather than an HPC resource manager.

## Round 2 — Two capacity partitions controlled by alpha/beta

**Model.** Manage L1 HBM and L2 Host DRAM as two protected partitions:

- `L1_budget(n,K) = alpha_n(K) * manageable_HBM_n`
- `L2_budget(n,K) = beta_n(K) * manageable_DRAM_n`

**What it fixed.** It turned backup into explicit capacity allocation and made the `K -> K+1` feedback loop understandable.

**Why it was insufficient.** Capacity feasibility was still confused with transfer feasibility. Increasing `alpha` or `beta` could create more reserved space without creating a safe copy window.

## Round 3 — Add a single copy-bandwidth guard

**Model.** Require a global `copy_bandwidth_token` in addition to `alpha/beta` capacity.

**What it fixed.** It acknowledged that backup consumes bandwidth and can hurt no-failure performance.

**Why it was rejected.** A single bandwidth number is a single-queue fallacy. Peer HBM, source D2H, PCIe, RDMA NIC, destination DRAM, and NVMe paths have different bottlenecks and can be independently idle or hot. It also could not represent sharing with KV prefetch/offload or collectives.

## Round 4 — Topology-aware resource graph

**Model.** Represent each candidate copy as a path through a graph:

`source HBM -> copy engine -> PCIe/NVLink -> NIC/RDMA -> destination HBM/DRAM -> optional NVMe`.

Each vertex/edge exposes capacity, queue, effective bandwidth, health, and failure domain.

**What it fixed.** Algorithm 2 could distinguish a destination with free space from a complete path with usable headroom. It also separated transfer route from redundancy grouping.

**Why it was insufficient.** A static graph still describes what resources exist, not when they are safely borrowable.

## Round 5 — Opportunity-window sensing

**Model.** Add an Idle Resource Sensor that emits short-lived opportunity windows from:

- inference phase: prefill, decode, idle, preemption, recovery;
- copy-engine `busy_until`, queue depth, and measured service rate;
- PCIe/NVLink/RDMA/NIC/NVMe utilization and headroom;
- HBM/DRAM/NVMe capacity watermarks;
- front-end prefetch/offload and collective reservations;
- path health and fail-slow anomaly confidence.

**What it fixed.** It imported REFT's key lesson—checkpoint work should exploit bubbles or otherwise idle resources—into online serving at a finer path level.

**Why it was insufficient.** It had no stable unit of admission and could oscillate by starting transfers that did not finish within an opportunity window.

## Round 6 — Chunked protection jobs with deadlines

**Model.** Convert a KV extent into preemptible protection jobs. Each job has:

`<request, version, byte range, conservative deadline, target fault domain, candidate paths>`.

Algorithm 2 admits only chunks that fit both capacity and the predicted path window. Completion is incremental.

**What it fixed.** It made backup resumable, preemptible, and compatible with fine-grained compute/communication overlap. It also let the safe frontier advance extent by extent.

**Why it was insufficient.** A pure earliest-deadline policy could starve large requests and consume every newly observed idle window.

## Round 7 — Credit-based multi-resource admission

**Model.** Every admitted chunk must atomically acquire:

1. an L1 or L2 capacity token under the current `alpha/beta` budget;
2. a token for each bottleneck resource along its path;
3. a fault-domain token proving the replica is independent of the protected source;
4. a no-failure SLO token proving that foreground headroom remains.

Credits are leases. Foreground traffic can revoke unstarted bandwidth leases; a started chunk stops only at a commit boundary.

**What it fixed.** It turned the design into a classic resource-management mechanism: classify, acquire, schedule, preempt, commit, and release.

**Why it was insufficient.** The allocator still lacked explicit failure-time correctness and garbage-collection semantics.

## Round 8 — Verified frontier and dependency-aware reclamation

**Model.** Use a per-request protection ledger:

`planned -> copying -> verified -> ACKed -> committed -> stale`.

Only the longest contiguous, version-matched, cross-fault-domain prefix is published as the safe frontier. Old extents are reclaimed only after a newer frontier supersedes them and all leases expire.

**What it fixed.** It separated asynchronous persistence from committed recoverability and made incremental copy safe.

**Why it was insufficient.** The fast-path policy could still drift under workload changes because `alpha/beta` and path credits were static.

## Round 9 — Windowed resource governor

**Model.** At the end of window `K`, seal:

- shadow demand, used capacity, and admission overflow per L1/L2;
- protection-deadline misses and committed useful bytes;
- per-path busy time, queue delay, fail-slow score, and revoked credits;
- no-failure TTFT/TPOT/P99, OOM, cache hit, and prefetch interference.

Apply a guarded state machine:

1. **Safety backoff:** if a no-failure guard is violated, shrink the responsible budget first.
2. **Path rebalance:** move credits away from hot or unhealthy paths to independent paths with headroom.
3. **Protection grow:** if urgent demand is missed and capacity plus path headroom exist, increase the corresponding `alpha` or `beta` by one step.
4. **Hold:** use hysteresis/cooldown inside a dead band.
5. **Canary publish:** atomically publish `budget_version=K+1`; rollback on guard failure.

**What it fixed.** It gave Algorithm 3 a simple, patent-friendly actuator rather than a large optimization formula.

**Why it was insufficient.** The model still described backup as a side service instead of a first-class shared system plane.

## Round 10 — Protection Plane

**Model.** Introduce a node-local Protection Agent and a cluster Protection Directory:

- the Router publishes only protection tickets and cached recovery hints;
- each source scheduler exposes execution phase and foreground reservations;
- each target Protection Agent owns Algorithm 2, local resource credits, transfers, ACKs, and Algorithm 3;
- the directory aggregates committed frontiers and fault-domain provenance without owning capacity;
- recovery reads the last committed frontier and never waits for a synchronous backup query.

**What it fixed.** It located mechanism ownership, data movement, and recovery authority. The system no longer looked like an application heuristic.

**Remaining concern.** It could overclaim that all runtime paths were already implemented.

## Round 11 — Evidence-bounded simulator/real-system split

**Model.** Separate three claims:

- **current SGLSim:** static alpha-beta path-duration prediction and aggregate overlap accounting;
- **proposed simulator extension:** topology/resource graph, queue/credit state, window sensing, and path-aware Algorithm 2;
- **real-system calibration:** microbenchmarked D2H/H2D, PCIe/NVLink/RDMA/NVMe contention and preemption points.

**What it fixed.** It makes the paper falsifiable and avoids calling duration accounting a deployed channel scheduler.

**Why it was not yet final.** It lacked a compact paper thesis.

## Round 12 — Final model: Slack-Harvesting Protection Fabric

**Paper thesis.** AegisKV treats KV backup as deadline-constrained, preemptible flow scheduling over a time-varying multi-resource graph. Application slack determines *when protection must become useful*; the system decides *where, through which channel, and in which idle window* to create it.

**Fast loop — Algorithm 2, per KV extent.**

1. Read the protection ticket, current `alpha/beta` budget version, resource graph, and opportunity windows.
2. Enumerate L1, L2, and Recompute plans. A backup plan includes destination, route, start window, rate cap, and fault domain.
3. Pick the earliest feasible plan that meets the conservative protection deadline while acquiring all path/capacity/SLO credits.
4. If no plan is feasible, downgrade L1 -> L2 -> defer -> Recompute. Never block the inference critical path.
5. Copy in chunks; verify and ACK each chunk; publish only the contiguous safe frontier.

**Slow loop — Algorithm 3, once per window.**

1. Seal demand, deadline miss, useful-copy, capacity, path, health, and no-failure SLO telemetry.
2. Apply safety backoff, path rebalance, protection grow, or hold.
3. Publish `alpha_n(K+1)` and `beta_n(K+1)` plus the next path-credit table under one version.
4. Activate at a safe barrier; canary and rollback if service guards regress.

**Window example used in the final figure.**

- Window `K`: `alpha=0.10`, `beta=0.30`. L1 has space but its peer path is busy; source D2H/PCIe has a short idle window; L2 Host DRAM has capacity.
- Request A, `Slack=8 ms`: L1 path misses the deadline, L2 path also misses, so its uncommitted suffix is Recompute.
- Request B, `Slack=45 ms`: a D2H -> PCIe -> remote Host DRAM window fits, so it receives L2 Backup.
- Request C, `Slack=180 ms`: backup benefit is below opportunity cost, so it is deliberately Recompute.
- Window `K` closes with urgent L1 misses, low L1 capacity pressure, saturated L2 DRAM, and a cool peer path expected in the next window.
- Window `K+1`: publish `alpha=0.16`, `beta=0.22`; shift capacity from L2 to L1 and issue credits on the recovered peer path. The same Algorithm 2 now admits more urgent L1 protection without changing its policy.

**Why this is the final choice.**

- It preserves the requested three-algorithm spine.
- It is a bottom-up HPC systems mechanism: topology, queues, overlap, bandwidth, capacity, preemption, consistency, and failure domains.
- It is explainable without a hard-to-protect objective function.
- It creates clear implementation and evaluation hooks.
- It keeps correctness independent of prediction and control quality.

## Round 13 — Evidence reset: peer HBM is not the default

**New evidence.** The focused 2025--2026 KV corpus found only one direct
fault-tolerant streaming-KV checkpoint paper, GhostServe. It uses GPUs to
compute parity but stores parity in Host memory; it explicitly rejects a
persistent peer-HBM copy as the general design. The other 29 papers provide
tiering, offload, restoration, caching, compression, and scheduling evidence
but not a fault-tolerance contract.

**Decision.** Reject the old main path `L1 peer HBM -> L2 Host`. Keep peer HBM
only as a gray emergency option for spare-HBM, fast-link, soft-GPU-only,
ultra-low-RTO cases.

## Round 14 — Unified Host pool and alpha/beta redesign

**Model.** Use one physical Host pool with typed logical leases:

`CACHE / LOCAL_PROTECT / REMOTE_SOURCE / REMOTE_REPLICA / PARITY /
TRANSFER_BUFFER`.

Redefine:

- `alpha`: protected-capacity share of the unified Host pool;
- `beta_p`: foreground-safe share of each transfer path.

An existing Host offload/cache page is promoted by metadata rather than copied
again. Local Host covers GPU/process failure; node/rack scope requires remote
replica/parity.

**Why it improved the system story.** Capacity, path bandwidth, eviction,
failure domain, and lifecycle are now concrete managed resources. The diagram
also connects fault tolerance to the existing CPU offload cache instead of
building a parallel memory subsystem.

## Round 15 — Correct Slack/RTO semantics

**Adversarial finding.** The previous rounds treated Algorithm 1 Slack as the
deadline of a copy started now. That is semantically wrong because failure time
is unknown. Slack constrains failure-time recovery:

`detect + reroute + restore + replay(dirty suffix) <= Slack`.

**Correction.** Maintain produced and group-committed frontiers. Derive a safe
dirty-suffix bound and a future protection-job expiry from KV growth. If no
recovery action fits the original Slack, return
`RECOVERY_SLO_INFEASIBLE`; do not call recompute safe.

## Round 16 — Dual-path certificate

**Adversarial finding.** A feasible write path does not imply feasible
recovery. The replacement GPU, model, communicator, routing state, remote
read/H2D, or parity decoder may be unavailable after failure.

**Correction.** Every Algorithm 2 action now needs:

1. a normal-time commit path before the derived dirty-frontier expiry;
2. a degraded-topology restore/recompute path with replacement compute that
   meets the original recovery budget.

This makes backup placement and failover capacity one resource-management
decision.

## Round 17 — Scope-specific group commit

**Live-code finding.** SGLang's current storage worker can break after a failed
page batch and still enqueue the operation on `ack_backup_queue`; the decode
offload manager then releases Host pages without checking the expected/success
page counts. The queue is an operation-completion notification, not a durable
commit.

**Correction.** A proposed commit contains expected and successful pages,
checksum, generation, topology epoch, failure-domain ACK/quorum/parity set, and
a TP/PP group-vector KV frontier. The emitted-output frontier remains separate.
Budget shrink grandfathers committed leases or migrates them through
`new commit -> old reclaim`.

## Round 18 — Final model: Recovery-Aware KV Resource Manager

The final mechanism chain is:

`recovery ticket -> dirty-frontier expiry -> exact Host lookup -> zero-copy
lease promotion or remote materialization -> atomic alpha/beta/fault-domain
leases -> scope-specific group commit -> dual restore/recompute certificate ->
cause-coded K feedback -> K+1 canary budget`.

This round replaces Round 12 as the design authority. Its full specification,
parallelism analysis, profiler evidence, implementation delta, and evaluation
contract are in
[`final_modeling_report.md`](final_modeling_report.md).

## Round 19 — Process ownership and three frontiers

**Independent-review finding.** The live
`DecodeKVCacheOffloadManager` and pinned Host pool are owned by the Scheduler
process. That state can survive GPU/device-state loss, but a real Scheduler
exit destroys the allocation. “Local Host covers process failure” was
therefore an overclaim.

**Correction.** Process scope requires a process-external Host-pool
owner/shared-memory daemon plus independently durable metadata, or a remote
domain. The dirty frontier is also defined over three distinct values:

- `K`: request-valid materialized-KV frontier;
- `C`: scope-protected/group-committed KV frontier;
- `E`: emitted-output frontier.

Only `dirty=K-C`; `E` remains separate for exactly-once output recovery.

## Round 20 — Post-failure decode quorum and logical topology

**Independent-review finding.** One parity shard on an independent node does
not recover data whose original shards all disappeared with the source node.
Likewise, a minimum over every physical TP rank incorrectly treats replicated
KV heads as conjunctive state, and ordinary “DP is spare compute” does not
describe DP-attention.

**Correction.** Node/rack scope commits either a full replica outside the
failed domain or a coded placement whose fragments still satisfy the decode
quorum after that declared failure. Protection is defined over logical
attention-TP shards under effective `T_a`; replicated physical heads are
alternatives. Standard full-replica DP and DP-attention now have separate
recovery contracts. PP anchors must cover the same token extent outside the
failed stage domain, and EP recovery includes placement, weights, routing, and
collective reconstruction.

## Round 21 — Manifest durability and final window semantics

**Architecture-review finding.** Payload ACKs are not sufficient if the only
manifest/directory disappears in the same failure. An early figure revision
also published `vK+1` without visibly showing who consumed it.

**Correction.** The logical transaction is now:

`payload/decode quorum + scope-durable manifest ACK -> group commit`.

The manifest is quorum-replicated outside the declared scope or represented by
co-stored self-describing commit markers; the lookup directory is rebuildable.
Payload without a durable manifest and manifest without surviving payload are
both uncommitted. The final figure explicitly states that Window `K+1` runs the
same Algorithm 2 with budget epoch `vK+1`, changing only its feasible resource
set. This is the final reviewed model.
