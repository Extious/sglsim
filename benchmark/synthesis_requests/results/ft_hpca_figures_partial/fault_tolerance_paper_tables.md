# Fault Tolerance Paper Tables

## Strategy Summary

| Metric | Value |
|---|---:|
| Recompute reduction | 50.1% |
| Critical-agent recompute reduction | 100.0% |
| GPU backup KV reuse | 50.1% |
| Output throughput change | 10.0% |
| Mean e2e latency change | -9.2% |
| Mean failover overhead change | -10.4% |

## Backup Lag

| Lag tokens | KV reuse (%) | Recompute tokens | Backed-up tokens |
|---:|---:|---:|---:|
| 0 | 26.0 | 86299.0 | 30305.0 |
| 1 | 0.0 | 85943.0 | 0.0 |
| 4 | 26.0 | 86303.0 | 30304.0 |
| 8 | 26.0 | 86310.0 | 30298.0 |
| 16 | 49.9 | 43305.0 | 43051.0 |
| 32 | 49.9 | 43239.0 | 43082.0 |

## Critical-Agent Coverage

| Critical ratio | Critical reused | Critical recompute | Non-critical recompute |
|---:|---:|---:|---:|
