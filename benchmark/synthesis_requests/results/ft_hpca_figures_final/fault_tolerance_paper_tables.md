# Fault Tolerance Paper Tables

## Strategy Summary

| Metric | Value |
|---|---:|
| Recompute reduction | 57.5% |
| Critical-agent recompute reduction | 100.0% |
| GPU backup KV reuse | 20.0% |
| Output throughput change | 23.7% |
| Mean e2e latency change | -12.7% |
| Mean failover overhead change | -1.3% |

## Backup Lag

| Lag tokens | KV reuse (%) | Recompute tokens | Backed-up tokens |
|---:|---:|---:|---:|
| 0 | 15.2 | 25907.6 | 14687.8 |
| 1 | 30.0 | 34490.8 | 17281.2 |
| 4 | 5.2 | 25878.2 | 6060.8 |
| 8 | 5.2 | 17262.0 | 6059.6 |
| 16 | 10.0 | 8661.0 | 8610.2 |
| 32 | 10.0 | 8647.8 | 8616.4 |

## Critical-Agent Coverage

| Critical ratio | Critical reused | Critical recompute | Non-critical recompute |
|---:|---:|---:|---:|
| 0.00 | 0.0 | 0.0 | 25948.8 |
| 0.25 | 0.0 | 0.0 | 17271.0 |
| 0.50 | 0.0 | 0.0 | 17313.8 |
| 0.75 | 8642.2 | 1.6 | 8622.0 |
| 1.00 | 17261.4 | 3.2 | 0.0 |
