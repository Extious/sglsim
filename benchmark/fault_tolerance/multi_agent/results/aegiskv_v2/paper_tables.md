# Multi-Agent AegisKV Paper Tables

## Strategy Summary (F2 critical failure)

| Metric | Value |
|---|---:|
| Baseline recompute tokens | 73932.8 |
| NaiveBackup recompute reduction | 100.0% |
| AegisKV recompute reduction | 100.0% |
| NaiveBackup backed-up tokens | 491520.0 |
| AegisKV backed-up tokens | 360448.0 |
| AegisKV backup reduction vs Naive | 26.7% |
| AegisKV vs Naive recompute gap | 0.0% |
| WCT median (Baseline) | 58.0s |
| WCT median (NaiveBackup) | 33.9s |
| WCT median (AegisKV) | 33.7s |
| AegisKV KV reuse rate | 100.0% |

## AegisKV Recovery Paths

| Path | Count |
|---|---:|
| prefix_restore_suffix_replay | 19 |
