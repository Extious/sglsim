# System overview validation

Validation snapshot: 2026-08-04 Asia/Hong_Kong.

## Outcome

`system_overview.drawio` is a native, editable, one-page mxGraph diagram.  Its
container hierarchy, restrained palette, rounded cards, and orthogonal paths
follow the visual grammar of:

`/home/comp/zhanzhao/sglang/benchmark/aegiskv_mooncake/docs/mooncake-hicache-control-data-plane.drawio`

The old PP=2/RTX 4090 physical slice has been removed.  The overview shows the
requested Qwen3-32B deployment: four nodes, four A100-80GB PCIe GPUs per node,
DP=4, and PP=4.  To preserve the Mooncake reference's whitespace, it draws DP0
once and DP1--DP3 as a repeated worker template; each template collapses the
four identical stage cards into one explicit `PP0 -> PP1 -> PP2 -> PP3` strip.

## Content contract

- top control plane: Algorithm 1 computes Slack, Algorithm 2 selects L1/L2/L3
  protection, Algorithm 3 adjusts alpha/beta for K+1, followed by K/C/E safety;
- central router/protection broker: execution/protection ranks, versioned plan,
  ACK frontier, retry reroute, and D/U/O feedback;
- two representative worker containers: DP0 and the repeated DP1--DP3
  structure, each with a PP0--PP3 GPU pipeline, Host pool, and protection agent;
- explicit short data paths: L1/L2 transfer, broker-to-agent control, and the
  broker-to-restore route, each confined to its own gutter;
- failure/recovery plane: scope detection, hybrid restore, persistent manifest
  quorum, replay, and evidence boundary;
- Algorithm 3 ownership: one alpha per DP replica/PP bottleneck and one beta
  per node-shared Host pool.

## Automated geometry checks

The fail-closed validator currently reports:

- page size: 1800 x 1120;
- 30 editable vertices and 16 editable edges;
- edge labels: 0;
- edge-to-foreground-node collisions: 0;
- edge-to-edge intersections: 0;
- diagonal route segments: 0;
- sibling overlaps: 0;
- gutters narrower than 16 px: 0;
- child/container overflows: 0;
- fallback-font text overflow: 0 cards.

All vertices use `Noto Sans CJK SC`; all edges use orthogonal routing; shadows
and gradients are disabled.  The validator requires both representative worker
templates, the explicit four-stage pipeline, Host pools, L3, D/U/O, K/C/E, and
the DP=4/PP=4 deployment labels.  It also estimates text wrapping using a wide
font fallback and rejects any leaf card below 11 px or without enough height.

## Evidence boundary

The figure intentionally says `FORMAL EXPERIMENT RUNNING` while the 32K Real
CC DP4/PP4 experiment is incomplete.  It does not reuse the old PP=2 profiler
numbers.  Final numbers are read only from
`AegisKV/presentation/experiments/qwen3_32b_dp4_pp4_final_summary.json` when
that artifact exists and has `gate_pass=true`; `--require-measured` fails
closed until then.

`AegisKV.demo.build_qwen3_final_summary` writes that artifact only after the
five-policy primary suite validates, p90 WCT improves by at least 1%, paired
mean WCT improves, at least half of workflows win, restored fraction is at
least 10%, and one additional matched workload/failure profile also has
positive p90/mean improvement with nonzero KV reuse.

## Artifacts

- editable diagram: `../system_overview.drawio`;
- generator: `generate_system_overview.py`;
- validator: `validate_system_overview.py`;
- preview renderer: `render_system_overview_preview.py`;
- current preview:
  `/home/comp/zhanzhao/sglsim/tmp/drawio/aegiskv-system-overview/system_overview_preview.png`.
