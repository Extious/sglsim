# Table 1: State Vector

| Symbol | Meaning |
|---|---|
| `h_t` | Host backup hit / ACK ratio |
| `g_t` | GPUReplica ACK ratio |
| `q_t` | retry / scheduler queue pressure |
| `p2p_t` | P2P bandwidth pressure |
| `h2d_t` | H2D restore pressure |
| `d2h_wt_t` | fixed write-through D2H pressure |
| `host_vis_t` | fraction of ACKed Host KV visible to retry ranks |
| `w_t` | workflow criticality skew |
| `l_t` | lifecycle distribution |
| `loc_{p,t}` | peer KV locality for candidate rank `p` |
| `load_{p,t}` | peer queue/load pressure |
| `dup_t` | duplicate Host/GPUReplica protected bytes |
| `alpha_gpu_t` | peer GPU backup budget ratio |
| `r_t` | recent recovery success / saved WCT |
