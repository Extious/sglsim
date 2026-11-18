# Idea 6: Profiler-Calibrated Multi-Resource Backup Broker

## Claim

Treat every KV protection extent as a deadline-constrained job that must obtain
both Host capacity and bandwidth-path credits. Use Algorithm 3 to change those
two budgets across windows instead of solving a large optimization formula on
the request path.

This is the control-plane counterpart of Idea 1 and the recommended semantics
for `alpha` and `beta`:

- `alpha_n(K)`: protected-capacity share of node `n`'s unified Host pool;
- `beta_{n,p}(K)`: maximum protection share of path `p` (D2H, Host bus, RNIC,
  remote DRAM). The overview displays an aggregate `beta`, while the runtime
  keeps a vector.

## Algorithm 2: fast path

For every protection extent:

1. **Build job.** Read the failure-time recovery budget, derive the
   dirty-frontier expiry, and attach bytes, current location, criticality,
   failure scope, topology epoch, and parallelism manifest. The recovery budget
   is not used directly as a copy deadline.
2. **Enumerate bounded actions.** Recompute; zero-copy Host promotion; local
   Host copy; full remote replica; post-failure-decodable coding; emergency
   peer HBM.
3. **Dual feasibility gate.** Require (a) a protect/commit path that finishes
   before the derived dirty-frontier expiry and obtains an `alpha` capacity
   lease, every required `beta_p` path credit, a no-failure SLO token, and an
   independent-failure-domain target or coding quorum that survives the
   declared fault; and (b) a degraded-topology
   restore/recompute path with replacement compute/model/communicator capacity
   that can satisfy the original recovery budget.
4. **Schedule.** Earliest-deadline class first; preserve minimum service for
   large/best-effort jobs; foreground work can revoke an unstarted credit.
5. **Execute/commit.** Copy micro-chunks, verify version/checksum, collect the
   payload/fault-domain quorum, publish a scope-durable manifest (external
   quorum or co-stored self-describing commit markers), and only then advance
   the contiguous frontier. The lookup directory must be rebuildable.
6. **Degrade or escalate.** On write-path infeasibility use promotion ->
   remote/local alternative -> defer/recompute. If the failure-time path still
   exceeds the recovery budget, mark `RECOVERY_SLO_INFEASIBLE` and trigger
   admission/reroute/reserve-capacity escalation. Never block a live request
   waiting for a backup lease, and never call an infeasible fallback protected.

## Algorithm 3: slow guarded governor

At the end of window `K`, seal:

- protection demand, admission overflow, dirty frontier, useful ACKed bytes;
- Host `CACHE`/`PROTECT` occupancy, eviction and fragmentation;
- D2H/Host/RNIC queues, achieved bandwidth, revoked credits, path health;
- foreground TTFT/TPOT/P99, throughput, OOM, cache-hit and prefetch latency;
- restore/recompute outcomes, replacement-target readiness, and fault-domain
  coverage;
- cause-coded misses: capacity, write path, failure domain, restore path,
  replacement target, or foreground guard.

Then apply an ordered state machine:

1. **Safety backoff:** if foreground SLO or OOM guard fails, reduce the
   responsible `beta_p` first; shrink `alpha` only for capacity pressure.
2. **Capacity grow/shrink:** high urgent misses plus safe memory headroom
   increases `alpha`; unused/stale protection decreases it.
3. **Path rebalance:** move credits away from hot/fail-slow paths; do not infer
   path headroom from GPU utilization alone.
4. **Hold:** deadband and cooldown suppress oscillation.
5. **Canary publish:** atomically publish `budget_version=K+1`; rollback on
   guard regression. A lower budget applies to new admission. Existing
   committed leases are grandfathered until TTL/dependency expiry, or migrated
   through `new commit -> old reclaim`; an in-flight extent stops only at its
   declared preemption boundary.

Per-node governors do not independently mint credits for a shared RNIC, remote
pool, or rack link. Each shared bottleneck has one admission owner that issues
sub-leases under the same group budget epoch.

## Concrete `K -> K+1` example

Window `K` publishes `alpha=0.10`, `beta=0.30`.

Observed:

- protected Host capacity reaches 97% and urgent jobs overflow;
- continuous D2H produces a decode-latency guard violation;
- existing Host cache pages have high promotion value;
- remote RNIC retains headroom.

Algorithm 3 publishes `K+1` as:

- `alpha=0.16`: keep more already-externalized pages under protection leases;
- aggregate `beta=0.18`: throttle new D2H work;
- per-path credits shift from D2H toward remote materialization of pages that
  are already in Host memory.

Algorithm 2 is unchanged. Only the feasible resource set changes.

## Real experimental evidence

The actual SGLang `profiler@fdebc938f7` run on Qwen3-0.6B/RTX4090 found:

| Metric | Prefill | Decode |
| --- | ---: | ---: |
| selected steps | 3 | 5 |
| median step | 24.013 ms | 22.819 ms |
| median kernel-union occupancy | 92.51% | 8.15% |
| observed gap p95 | 10.848 us | 161.985 us |

A continuous 256 MiB pinned D2H test, issued in 16 MiB synchronized chunks,
reached 25.416 GB/s. Decode median latency was 718.677 ms versus 677.124 ms
before and 652.764 ms after; relative to the mean of the two controls, the
increase was 8.08%. Prefill did not show stable slowdown, which is not evidence
that D2H is free.

The GPU was shared, samples were small, TP=1, and CUDA graphs were disabled.
These numbers are mechanism evidence, not publication-grade performance.
See the full [profiler report](evidence/sglang_profiler/REPORT.md).

## Design consequence

- prefill: default pause or extremely small control chunks;
- decode: candidate window only, using micro-chunks and an SLO feedback loop;
- do not convert profiler kernel gaps directly into copy credits;
- calibrate each path with an active interference probe and production
  telemetry;
- reserve foreground prefetch, offload, and collectives before issuing backup
  credits.

## Falsifiable experiment

Compare:

1. fixed bandwidth;
2. GPU-utilization gate;
3. trace-gap gate;
4. capacity-only `alpha`;
5. full `alpha + beta_p` broker.

Run continuous batching, CUDA graphs, and TP/PP/EP across PCIe/NVLink/RDMA
topologies. Measure protection goodput, foreground P99, missed deadlines,
credit revocations, oscillation, and per-path utilization.

The idea fails if path telemetry is too delayed/noisy to control interference,
if micro-chunk overhead removes useful bandwidth, or if canary rollback reacts
slower than SLO damage.

## Patent-friendly mechanism boundary

`sense -> seal telemetry -> publish capacity/path credits -> acquire leases ->
micro-chunk execution -> verify/ACK -> advance frontier -> guard/canary/rollback`.

This is an implementable resource-control protocol, not ownership of a
particular scoring equation.
