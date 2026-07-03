# DP=2 SGLang Long Failure Smoke Job 32978

Status: success, longer-output control-path evidence.

Remote run:

- Host: `hkbugpusrv11`
- Slurm job: `32978`
- SGLang repo: `/home/comp/25482351/sglang`
- Branch/SHA: `multi-level-backup` / `360072905d9373a33fb96360474c9e6e1682f0ce`
- Model: `Qwen/Qwen3-8B`
- Topology: DP=2, TP=1, A100 x2
- Workload: 4 routed concurrent requests, `max_tokens=512`, failure delay 8 s
- Server controls: HiCache write-through, CUDA graph disabled, piecewise CUDA graph disabled, overlap schedule disabled for control-path smoke
- Result directory: `results/dp2_failure_smoke_32978/`

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
| p95 latency | 16.893 s |
| p99 latency | 16.893 s |
| Response pre-failover output tokens | 760 |
| Response pre-failover backed-up tokens | 0 |

Interpretation:

- Compared with job `32977`, this longer run triggered failure after each
  in-flight request had generated 190 output tokens. SGLang still captured and
  re-dispatched all 4 requests, and every final response carried
  `sglext.failover`.
- The backed-up-token count remained 0. This is important negative evidence:
  the current SGLang small-scale smoke validates failover continuity, but does
  not yet expose an ACKed HostBackup frontier usable for Host-visible restore
  claims.
- Final HPCA evidence still needs a paired SGLang run where recoverable
  Host/GPUReplica tokens are nonzero and where Host-visible, Host-invisible,
  and peer-copy behavior can be compared under matched normal-path cost.

Artifacts:

- `results/dp2_failure_smoke_32978/failure_summary_checked.json`
- `results/dp2_failure_smoke_32978/failure_summary_checked.csv`
- `results/dp2_failure_smoke_32978/candidate_fig_failure_smoke_sglang_checked.csv`
- `results/dp2_failure_smoke_32978/client_results.jsonl`
- `results/dp2_failure_smoke_32978/failover_events.jsonl`
- `results/dp2_failure_smoke_32978/server.log`
