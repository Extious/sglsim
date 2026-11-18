# DP=2 SGLang Remote-Backup Failure Smoke Job 32980

Status: success, storage-backed Host-visible frontier smoke.

Remote run:

- Host: `hkbugpusrv11`
- Slurm job: `32980`
- SGLang repo: `/home/comp/25482351/sglang`
- Branch/SHA: `multi-level-backup` / `360072905d9373a33fb96360474c9e6e1682f0ce`
- Model: `Qwen/Qwen3-8B`
- Topology: DP=2, TP=1, A100 x2
- Workload: 4 routed concurrent requests, `max_tokens=512`, failure delay 8 s
- Server controls: HiCache write-through, `hicache_storage_backend=remote_backup`,
  centralized remote backup server, CUDA graph disabled, piecewise CUDA graph
  disabled, overlap schedule disabled for control-path smoke
- Result directory: `results/dp2_failure_smoke_32980/`

Remote-backup setup:

- The Slurm script launched
  `sglang/srt/mem_cache/storage/remote_backup/launch_remote_backup_server.py`
  before the SGLang server.
- SGLang workers used
  `--hicache-storage-backend remote_backup` and
  `--hicache-storage-backend-extra-config '{"remote_backup_mode":"client","remote_backup_url":"localhost:33180","remote_backup_buffer_size_gb":16}'`.
- The remote backup server log confirms TCP backup writes and prefix-match
  queries from both DP ranks.

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
| p95 latency | 18.715 s |
| p99 latency | 18.715 s |
| Response pre-failover output tokens | 648 |
| Response pre-failover backed-up tokens | 5132 |

Interpretation:

- This is the first SGLang small-scale run in this artifact set where the
  failover response reports a nonzero recoverable frontier:
  1,283 backed-up tokens per request, 5,132 tokens total.
- It validates the real control path and the response surface for a
  storage-backed Host-visible frontier: `/simulate_gpu_failure` captures active
  requests, failover dispatch carries `pre_failover_backed_up_tokens`, retry
  completes on the surviving rank, and final OpenAI-compatible responses carry
  `sglext.failover`.
- It is still a smoke, not final Fig. 9/11b evidence. The run disables overlap
  schedule to avoid node JIT issues, uses only four requests, and does not yet
  compare Host-visible, Host-invisible, and peer-copy policies under a paired
  workload.

Artifacts:

- `results/dp2_failure_smoke_32980/failure_summary_checked.json`
- `results/dp2_failure_smoke_32980/failure_summary_checked.csv`
- `results/dp2_failure_smoke_32980/candidate_fig_failure_smoke_sglang_checked.csv`
- `results/dp2_failure_smoke_32980/client_results.jsonl`
- `results/dp2_failure_smoke_32980/failover_events.jsonl`
- `results/dp2_failure_smoke_32980/remote_backup_server.log`
- `results/dp2_failure_smoke_32980/server.log`
