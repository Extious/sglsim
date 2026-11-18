# DP=2 SGLang Failure Smoke Job 32977

Status: success, small-scale control-path evidence only.

Remote run:

- Host: `hkbugpusrv11`
- Slurm job: `32977`
- SGLang repo: `/home/comp/25482351/sglang`
- Branch/SHA: `multi-level-backup` / `360072905d9373a33fb96360474c9e6e1682f0ce`
- Model: `Qwen/Qwen3-8B`
- Topology: DP=2, TP=1, A100 x2
- Server controls: HiCache write-through, CUDA graph disabled, piecewise CUDA graph disabled, overlap schedule disabled for control-path smoke
- Result directory: `results/dp2_failure_smoke_32977/`

Why overlap was disabled:

- Job `32972` on `hkbugpusrv07` failed while compiling the SGLang JIT
  `resolve_future_token_ids` kernel because the node compiler runtime lacked
  the C++ `<version>` header.
- Job `32976` on `hkbugpusrv11` then showed the direct Python launch needed the
  SGLang virtualenv bin directory on `PATH` so JIT subprocesses could find
  executable `ninja`.
- The final job `32977` used `${SGLANG_DIR}/.venv/bin/python3`, prepended
  `${SGLANG_DIR}/.venv/bin` to `PATH`, excluded `hkbugpusrv07`, and disabled
  overlap schedule. This keeps the artifact scoped to failure-control-path
  evidence, not throughput evidence.

Checked summary from raw `client_results.jsonl` and `failover_events.jsonl`:

| Metric | Value |
|---|---:|
| Requests | 4 |
| Successful responses | 4 |
| Failure API status | 200 |
| Recovery API status | 200 |
| Captured requests | 4 |
| Re-dispatched requests | 4 |
| Responses with `sglext.failover` | 4 |
| Event rows | 7 |
| `gpu_failure_simulated` events | 2 |
| `failover_dispatched` events | 2 |
| `resume_accepted` events | 2 |
| `gpu_recovered` events | 1 |
| p95 latency | 11.194 s |
| p99 latency | 11.194 s |
| Response pre-failover output tokens | 8 |
| Response pre-failover backed-up tokens | 0 |

Interpretation:

- This validates the real SGLang HTTP control path for in-flight DP-rank
  failure: `/simulate_gpu_failure` captures active requests, sends a failover
  batch through tokenizer manager, re-dispatches them to the surviving rank,
  and returns completed OpenAI-compatible responses with `sglext.failover`.
- The run does not validate AegisKV's final physical Host/GPUReplica recovery
  benefit. `pre_failover_backed_up_tokens=0`, so this is a retry-continuity and
  response-surface smoke, not a Host-visible restore or peer-copy measurement.

Artifacts:

- `results/dp2_failure_smoke_32977/failure_summary_checked.json`
- `results/dp2_failure_smoke_32977/failure_summary_checked.csv`
- `results/dp2_failure_smoke_32977/candidate_fig_failure_smoke_sglang_checked.csv`
- `results/dp2_failure_smoke_32977/client_results.jsonl`
- `results/dp2_failure_smoke_32977/failover_events.jsonl`
- `results/dp2_failure_smoke_32977/server.log`
