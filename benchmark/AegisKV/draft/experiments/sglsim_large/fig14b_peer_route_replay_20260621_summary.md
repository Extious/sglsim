# Fig. 14b Peer-Route Replay Derivation

- run_id: `sglsim-fig14b-peer-route-dp4-r3-20260621`
- matched recovery rows: `72`
- DP size: `4`
- random healthy route hit probability: `0.3333`
- baseline p99 recompute overhead: `7.3480 ms`

| policy | normalized WCT | p99 overhead vs recompute | mean restore overhead |
|---|---:|---:|---:|
| non-peer route miss | 1.0000 | 100.00% | 6.0384 ms |
| random healthy route | 0.9928 | 90.24% | 5.3456 ms |
| AegisKV peer route | 0.9783 | 70.72% | 3.9600 ms |

This is a matched-request replay derivation, not a final physical P2P KV-copy policy sweep.
