# AegisKV deep research synthesis

> **Superseded design record.** This file preserves the pre-profiler,
> peer-HBM-centric synthesis for audit history. Its direct-Slack-as-copy-deadline
> assumption and `alpha=L1 peer HBM / beta=L2 Host` semantics were rejected
> after the 60-paper review, live SGLang source audit, profiler experiment, and
> adversarial architecture review. The current design authority is
> [`final_modeling_report.md`](final_modeling_report.md), and the editable
> current figure is `../system_overview.drawio`.

## Auditable corpus

The accompanying [`paper_matrix_2025_2026.csv`](paper_matrix_2025_2026.csv) contains exactly 100 papers from the official 2025–2026 OSDI, NSDI, USENIX ATC, and FAST presentation pages:

| Direction | Count | Primary question |
|---|---:|---|
| Resource allocation | 20 | How should capacity, bandwidth, queues, and time-varying idle resources be represented and allocated? |
| Distributed fault tolerance | 20 | What makes an asynchronous replica safe, useful, and independent at failure time? |
| LLM inference | 20 | Where do KV movement, tiering, prefetch, and serving SLOs contend for the same hardware? |
| Training/checkpointing | 20 | How can checkpoint work exploit bubbles, redundancy, existing data flows, and reconfigurable layouts? |
| Multi-agent/workflow scheduling | 20 | How should dynamic DAGs, long tails, and heterogeneous stages create or consume opportunity windows? |

The corpus has 60 papers from 2025 and 40 from 2026. Every row records an official URL, a mechanism readout, the resource/failure signal it uses, and one bounded implication for AegisKV. The matrix is a targeted design corpus, not a bibliometric claim that these are the only relevant papers.

## Direct audit of REFT

The local REFT paper was read and pages 3–8 were rendered and visually inspected:

[`Fault-Tolerant Hybrid-Parallel Training at Scale with Reliable and Efficient In-memory Checkpointing.pdf`](/home/comp/zhanzhao/DeepSpeed/REFT/Fault-Tolerant%20Hybrid-Parallel%20Training%20at%20Scale%20with%20Reliable%20and%20Efficient%20In-memory%20Checkpointing.pdf)

The transferable mechanisms are:

1. **Resource competition is path-specific.** Figure 2 separates host coordination from host-device coordination: asynchronous work can still block the training communication path.
2. **Hierarchical opportunity use.** Figure 4 places snapshots first in pipeline bubbles, then alongside computation, and only then alongside communication when training and snapshot use different interconnects.
3. **Bubble-sized partitioning.** Figure 5 / Algorithm 1 estimates snapshot time and bubble time, putting the portion that fits into the bubble there and deferring the remainder to a more intrusive layer.
4. **Interconnect separation.** The Layer-3 design exploits training over NVLink and snapshot over PCIe. “Asynchronous” is not sufficient; the data path must be independent or have measured headroom.
5. **Protection is distinct from snapshot creation.** ARC/AEC/AOR protect an in-memory snapshot against node loss; the location and redundancy mechanism are selected separately from when the snapshot is produced.
6. **Committed-state management.** REFT retains completed and ongoing copies and uses version/state-machine logic at restart.

AegisKV should adopt the scheduling principle, not copy the training-specific algorithm. Serving has finer, irregular prefill/decode phases, per-request recovery deadlines, shared KV offload/prefetch traffic, and no global iteration barrier.

## What the live SGLSim currently models

The current checkout provides useful duration accounting but not a resource-aware protection scheduler:

- [`communication/types.py`](/home/comp/zhanzhao/sglsim/python/sglsim/srt/communication/types.py:7) exposes only `H2D`, `D2H`, `DISK_READ`, and `DISK_WRITE`. Requests carry page IDs and page size, not endpoints, link identity, queue, stream, or fault domain.
- [`communication/predictor.py`](/home/comp/zhanzhao/sglsim/python/sglsim/srt/communication/predictor.py:70) chooses a static path formula. H2D and D2H read platform memory bandwidths; disk I/O reads disk bandwidth. It records aggregate durations but has no shared-link arbitration.
- [`mem_cache/hicache.py`](/home/comp/zhanzhao/sglsim/python/sglsim/srt/mem_cache/hicache.py:38) keeps write/load/L3 queues but merges D2H and L3 write time into one scalar `pending_backup_dur`. [`consume_backup_overlap()`](/home/comp/zhanzhao/sglsim/python/sglsim/srt/mem_cache/hicache.py:122) subtracts forward duration from that scalar.
- [`pipeline_accounting.py`](/home/comp/zhanzhao/sglsim/python/sglsim/srt/simulation/pipeline_accounting.py:32) treats D2H as hidden behind the current forward and H2D as hidden behind the previous forward when overlap is enabled. This is a wall-clock model, not proof that a specific copy engine or link was idle.
- [`aegis_l1_l2.py`](/home/comp/zhanzhao/sglsim/python/sglsim/srt/managers/aegis_l1_l2.py:200) correctly interprets `alpha` and `beta` as L1/L2 capacity budgets. Its per-chunk admission uses value density and capacity replacement, and [`close_window()`](/home/comp/zhanzhao/sglsim/python/sglsim/srt/managers/aegis_l1_l2.py:499) updates the two ratios from demand/used/overflow and normal pressure. It does not reserve an actual transfer route or opportunity window.

Therefore the final figure labels topology sensing, path credits, and window scheduling as the **proposed Protection Fabric**, not as already implemented runtime behavior.

## Cross-direction synthesis

### 1. “Idle” must be resource- and dependency-specific

Training bubble systems, chunk-level overlap, and production idle analyses agree on one point: aggregate utilization is an unsafe admission signal. A GPU may be idle while waiting on the NIC; that interval is not free for another NIC transfer. Conversely, compute may be saturated while an independent PCIe copy engine is usable. The sensor must emit a vector:

`<resource, interval, residual capacity, foreground reservation, health confidence>`.

### 2. Capacity and transfer feasibility are different allocations

`alpha` and `beta` remain capacity partitions because that meaning is clear and already implemented. A separate path-credit table represents time/bandwidth. An extent is admitted only if both allocations succeed atomically. This follows multi-resource storage allocation and avoids the “space is free, therefore backup is feasible” mistake.

### 3. The schedulable unit is a preemptible protection extent

The corpus repeatedly favors finer units: granular compute, chunk-centric overlap, sparse checkpointing, token-level co-serving, micro-flow RL scheduling, and iteration-level preemption. AegisKV should use page-aligned extents that can stop at an ACK boundary. This turns a multi-GB request into a sequence of bounded jobs and lets the safe frontier grow incrementally.

### 4. The scheduler chooses route and time, not merely tier

For each candidate:

- `L1`: source HBM -> GPU copy/transport path -> peer HBM in an independent failure domain;
- `L2`: source HBM -> D2H copy engine -> PCIe -> Host DRAM, optionally across RNIC to the selected backup rank;
- `Recompute`: no immediate copy, but a future compute job whose cost depends on available prefix and serving load.

The plan includes destination, path, earliest start, rate cap, deadline, and expected committed frontier. “L1/L2” describes the endpoint; it does not identify the bottleneck.

### 5. Foreground reservations dominate backup opportunities

KV prefetch/offload, attention offload, model loading, collectives, and request migration share the same HBM, PCIe, NIC, and DRAM resources. The front end first publishes reservations. Protection receives revocable residual credits. This is how the design prevents a backup optimization from becoming a no-failure regression.

### 6. Recovery correctness is an ACKed-prefix invariant

The control loop may mispredict without corrupting recovery. Only a contiguous, version-matched, checksum-verified extent in an independent fault domain advances the safe frontier. A failed or stale copy has zero recovery value and is never discounted at retry time.

### 7. Two timescales avoid global optimization on the request path

The online loop performs bounded candidate enumeration and credit acquisition. The window loop changes supply. It never recomputes decisions already made in `K`, preventing same-window circular causality:

`Alg.3(K) publishes budget -> Alg.2(K) consumes it -> sealed feedback -> Alg.3(K+1)`.

## Final paper idea

### Working title

**AegisKV: Slack-Harvesting Protection over Idle Resource Windows for Fault-Tolerant LLM Serving**

### One-sentence thesis

AegisKV converts request recovery slack into deadlines for preemptible KV protection flows, then schedules those flows over topology-aware idle windows while a guarded per-node controller reallocates L1/L2 protection capacity across windows.

### System components

1. **Protection Ticket Publisher (Algorithm 1).** Emits conservative slack, criticality, request/version, KV range, and failure scope.
2. **Resource Opportunity Sensor.** Observes phase, copy-engine state, PCIe/NVLink/RDMA/NIC/NVMe queues, effective bandwidth, capacity, front-end reservations, and fail-slow health.
3. **Protection Graph.** Represents endpoints, bottleneck resources, failure domains, and short-lived path credits.
4. **Slack-Harvesting Allocator (Algorithm 2).** Chooses L1/L2/Recompute plus route/window/rate; supports downgrade, defer, replacement, and preemption at extent boundaries.
5. **Transfer and Commit Engine.** Executes asynchronous chunks and advances only the verified ACK frontier.
6. **Windowed Resource Governor (Algorithm 3).** Applies safety backoff, path rebalance, protection growth, hysteresis, canary publication, and rollback.
7. **Recovery Directory.** Stores compact committed-frontier provenance and serves failure-time recovery without a synchronous backup query.

### Why it is a systems paper rather than an application

- It exposes and arbitrates concrete hardware resources: HBM, Host DRAM, copy engines, PCIe, NVLink/RDMA, NIC queues, and NVMe.
- It models computation/communication overlap as reservations on a time-varying resource graph.
- It defines ownership and protocols across Router, Scheduler, Protection Agent, Directory, and recovery path.
- It states safety and liveness boundaries independent of the application score.
- Its central evaluation is resource interference and recoverability, not only workflow completion time.

### Patent-friendly mechanism chain

The claimable implementation sequence is:

`sense -> publish credits -> classify protection ticket -> acquire multi-resource lease -> route and schedule extent -> verify and ACK -> advance frontier -> seal feedback -> actuate alpha/beta -> canary/rollback`.

The novelty is expressed as cooperating components and state transitions. Formulas are calibration aids, not the invention itself.

## K to K+1 example

### Window K

- Published: `alpha(K)=0.10`, `beta(K)=0.30`, `budget_version=K`.
- L1 peer HBM has free reserved space, but the peer path is busy with foreground communication.
- Source D2H/PCIe has a 28 ms opportunity, and the chosen L2 Host DRAM domain has headroom.
- Request A (`Slack=8 ms`) cannot obtain a complete safe path; its uncommitted suffix remains Recompute.
- Request B (`Slack=45 ms`) obtains L2 capacity and a D2H/PCIe/RDMA path lease; the copy finishes and ACKs before its deadline.
- Request C (`Slack=180 ms`) has low recovery urgency and is intentionally left to Recompute.

### Sealed feedback

- high urgent L1 miss bytes;
- low L1 capacity pressure;
- high L2 capacity/DRAM pressure;
- cool peer path forecast after foreground collective completion;
- no TTFT/TPOT/P99 or OOM violation.

### Window K+1

- Publish atomically: `alpha(K+1)=0.16`, `beta(K+1)=0.22`, `budget_version=K+1`.
- Reclaim stale L2 extents, expand L1 protection only at an allocation barrier, and issue peer-path credits only after the foreground reservation ends.
- Algorithm 2 is unchanged. Its feasible set changes: more urgent extents can now enter L1, while non-urgent L2 demand is downgraded or deferred.

## Required implementation delta

The paper mechanism needs explicit new state; it cannot be implemented by only retuning the existing communication predictor.

1. Extend `CommunicationRequest` with source/destination, link/resource IDs, stream class, priority, deadline, and failure domain.
2. Add a topology/resource registry with per-resource queues, `busy_until`, measured service rates, reservations, and health.
3. Replace scalar `pending_backup_dur` with per-path, per-chunk transfer state.
4. Add foreground reservations from H2D load-back, D2H write-through, KV prefetch/offload, and collectives.
5. Add credit leases and preemption at chunk boundaries.
6. Extend Algorithm 2 output from `{tier, tokens}` to `{tier, destination, path, window, rate, extent, deadline}`.
7. Extend Algorithm 3 input with per-path demand/miss/interference and publish a path-credit table under the same budget version as `alpha/beta`.
8. Persist the transfer state machine and ACKed contiguous frontier in structured result artifacts.

## Evaluation contract

### Baselines

- No Backup / Recompute.
- Fixed L2 HostBackup.
- Fixed L1 Backup.
- Current capacity-only `alpha/beta` AegisKV.
- Path-aware without opportunity sensing.
- Opportunity-aware with static `alpha/beta`.
- Full Slack-Harvesting Protection Fabric.

### Workloads and platforms

- single-turn, multi-turn, and agentic DAG workloads;
- fixed and heavy-tailed input/output lengths;
- topology cases with shared versus independent copy/communication paths;
- heterogeneous ranks and injected fail-slow links;
- real D2H/H2D/RDMA/NVMe microbenchmarks plus calibrated SGLSim scale studies.

### Primary metrics

- no-failure TTFT, TPOT, P99, throughput, and OOM;
- per-resource utilization and queue delay;
- backup goodput versus raw bytes copied;
- percentage of bytes hidden inside valid opportunity windows;
- protection deadline miss rate by request class;
- committed safe-frontier coverage;
- failure-time restore, recompute, and end-to-end WCT;
- controller convergence, oscillation, credit revocation, and rollback rate.

### Critical ablations

- no link identity;
- no copy-engine sensing;
- aggregate utilization instead of dependency-aware idle;
- no foreground reservation;
- no path health/fail-slow signal;
- whole-request copy instead of chunking;
- no ACK frontier;
- scalar global `alpha/beta` instead of per-node budgets;
- same-window feedback instead of `K -> K+1`;
- formula optimizer versus guarded state machine.

## Evidence boundary

The 100-paper matrix and REFT audit support the design choices. They do not constitute evaluation evidence for AegisKV. The current draw.io is a proposed system overview. A paper claim becomes “implemented” only after the resource graph, queue/credit state, path-aware Algorithm 2, windowed Algorithm 3 inputs, and structured event traces exist and are validated against real transfer measurements.
