# AegisKV complete current-design metadata simulation

Status: **implemented and executable**. This is the unified SGLSim path for
the complete `source_owned_independent_hbm_dram_v2` control and failure
semantics. It supersedes neither the immutable `sglsim_large_scale_v2`
lineage nor the preliminary WCT figures; it closes the mechanisms that those
campaigns explicitly did not model.

## What “complete” means here

One continuous-time event stream executes and audits:

1. causal visible-residual-DAG Slack updates;
2. progressive prefix/decode chunk sealing;
3. independent SlackLandlord admission into a real per-worker HBM pool and a
   shared Store-DRAM pool;
4. independent source-owned D2D and D2H queues, ACKs, and
   `PENDING/COMMITTED/ABORTED` state;
5. per-target HBMQuotaAIMD;
6. target-first failure with an unchanged route version, old-target HBM
   invalidation, DRAM preservation, and copy-ahead against the replacement's
   actual capacity pool;
7. source fail-stop, pending-copy abort, output fence, replacement delay,
   remaining-Slack recovery ordering, committed-prefix-only restore, actual
   load-byte accounting, all-shard activation, and silent replay;
8. queue outage and rescheduling for unstarted work on the failed source;
9. contiguous route-version rebind, new source-owned reprotection, repair
   without resurrection of old HBM, and repeated continuous-time faults.

The implementation is
`python/sglsim/srt/simulation/aegis_complete.py`. Its fail-closed auditor
requires the target-first, source-failure, recovery, repair, and queue-outage
events to be exercised before returning `PASS`.

## Run

```bash
uv run python AegisKV/experiments/sglsim_complete_v1/run.py
uv run --extra test pytest test/unit/srt/test_aegis_complete.py -q
```

The demo writes immutable `result.json`, `audit.json`, and `summary.json` under
`results/complete-demo-v1/`. The scenario deliberately places two requests on
one source, fails their bound HBM target first, then fails the source. It
therefore exercises both the active recovery path and queued-request outage.

## Evidence boundary

This is a complete **metadata/control-plane DES**, not a physical result.
Copy, load, recompute, replacement, and replay durations are explicit
calibration inputs. SGLSim does not move physical KV bytes or execute
CUDA/NCCL/Mooncake transport, so `physical_result=false` remains mandatory.

Large-scale statistical conclusions still require a new preregistered
campaign built on this runner. Historical simulation and preliminary WCT
numbers must not be relabeled as results from the complete path.
