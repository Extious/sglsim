# G1/G2 causal observability smoke

This experiment checks whether one opt-in SGLSim run can capture the state that
Algorithm 2 is allowed to observe at a failure event.  It does not test a real
GPU failure, actual KV movement, prediction accuracy, or workflow benefit.

The runtime hook is observational and only active with
`scheduler_telemetry=True`.  Each scheduler row may contain:

- active request/workflow/agent identity and queue location;
- admission-known prompt/decode budget and currently observed progress;
- currently committed/cache/restore token counts;
- L1/L2 capacity and headroom;
- exact DP/PP/TP stage identity.

The snapshot builder chooses the latest row for each physical stage whose
timestamp is no later than an accepted failure-control event.  It rejects keys
that encode final completion, actual output, trace-oracle Slack, or other future
state.  It also replays versioned backup ACK, eviction, supersede, and failed-
target events.  Post-hoc outcomes are stored outside the causal snapshot.

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=python uv run python \
  AegisKV/experiments/g1_observability/run_snapshot_smoke.py \
  --output-dir AegisKV/experiments/g1_observability/results
```

Expected evidence class: `sglsim_online_instrumentation_smoke`.  Passing means
only that domain, queue, capacity, victim and bystander identities can be joined
without future leakage, and that stale or unmaterialized copies are rejected.

The current runtime keeps admission-time reservation and recovery publication
separate.  The 224-token request budget first produces a
`metadata_reservation_only` ACK.  After the source finishes a 64-token prompt
chunk, the aligned target emits a modeled GroupCommitAck carrying
`execution_id`, `block_generation`, and `source_committed_token_end=64`.
Snapshot v3 therefore reports one 64-token reusable prefix for the interrupted
request, never the 224-token future budget.  The latest checked-in snapshot is
`14f870634cf39b31b1ef222d3366171eeaed9fffb02db8bd9a9351406b5a2f43`.

This remains metadata-model evidence: SGLSim does not transfer or checksum real
KV tensors.  The failed-rank reload horizon is a configured causal model and is
reported as `MODELED`, not measured.  Consequently G1 and G3 remain unproven.

The paired mechanism smoke can be run with:

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=python uv run python \
  AegisKV/experiments/g1_observability/run_paired_bystander_smoke.py \
  --output-dir AegisKV/experiments/g1_observability/paired_results
```

It places identical healthy bystanders on two DP processes, then sends a
committed-prefix retry to one and a forced-recompute retry to the other.  It is
deliberately labelled `NOT_IDENTIFIED`: ranks differ and pre-failure progress is
not exactly matched.  Two repeated checked-in runs produced opposite signs for
both retry and bystander latency deltas.  The machine-readable
[`paired_repeatability_audit.json`](paired_repeatability_audit.json) therefore
sets `counterfactual_identification=FAILED`.  Formal G1 evidence requires
rank-crossover assignment, an identical replayed arrival/fault trace, matched
pre-failure state, multiple paired seeds with confidence intervals, and
hardware-calibrated or real-system outcomes.

Two fail-closed Algorithm-2 input audits are retained deliberately:

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=python uv run python \
  AegisKV/experiments/g1_observability/audit_algorithm2_snapshot.py

UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=python uv run python \
  AegisKV/experiments/g1_observability/audit_recovery_laxity_snapshot.py
```

The first audits the older full online shadow contract.  The second audits the
smaller Slack-Deadline contract: causal Slack provenance and explicit
`need_state`/`slack_kind`, a versioned Router
target, causal target availability, a matching committed prefix, calibrated
protected/unprotected recovery service, layout compatibility, and foreground-
bounded resource caps.  The current snapshot is correctly `REFUSED_TO_SCORE`;
the smaller audit removes unidentifiable online bystander prediction but does
not fill any retained field with a default constant.  An `UNKNOWN` Slack ticket
is accepted only as the explicit fallback `slack=0, usable=false`; a positive
unknown Slack is rejected.
