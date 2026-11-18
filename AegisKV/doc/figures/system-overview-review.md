# AegisKV System Overview visual review

> **Historical review, superseded by the independent-tier design.** The
> reviewed Draw.io source and this record still describe a target-owned
> destage path. They are retained to explain the visual iteration, but they do
> not pass the current `source_owned_independent_hbm_dram_v2` semantic audit.
> See `../design/current_design.md` and `../PROJECT_STATUS.md` before reworking
> the diagram.

This record documents the visual comparisons and reviewer-style iterations for
`aegiskv-paper-system-design-v2.drawio`. The figure is an architecture diagram;
it contains no measured performance claim.

## Requirements

- Use neutral fills plus only two arrow colors.
- Use dark blue for control, metadata, commands, ACKs, pressure, commit, and
  failure signals.
- Use orange for physical KV tensor movement.
- Make the application, serving control, and KV data planes visually explicit.
- Keep type readable at the final two-column paper width.
- Show both the normal protection path and the failure recovery path without
  expanding the overview into an algorithm flowchart.

## Related-work visual baselines

- [Mooncake, Figure 2](https://www.usenix.org/system/files/fast25-qin.pdf)
  cleanly separates the Conductor, inference instances, and distributed KV
  pool. Its repeated P/D instances make the physical topology concrete, but the
  figure does not carry workflow criticality or a failure-control loop.
- [DejaVu, Figure 5](https://proceedings.mlr.press/v235/strati24a.html) makes
  cache exchange, replication, and swapping distinguishable with three arrow
  colors. AegisKV instead uses the requested two-class vocabulary: all KV bytes
  are orange and every non-KV signal is blue.
- [GhostServe, Figure 3](https://proceedings.mlsys.org/paper_files/paper/2026/file/d99e8e80a6c41e148db686918dd7eab3-Paper-Conference.pdf)
  is effective because checkpoint and recovery are direct physical arrows next
  to the protected state. AegisKV follows this principle with numbered P2P
  Peer-HBM protection, direct DRAM protection, target-owned destage, and Store
  restore paths.
- [Agentix, Figure 8](https://www.usenix.org/system/files/conference/nsdi26/nsdi26spring_luo_prepub.pdf)
  clearly carries program context from the frontend into the serving engine.
  AegisKV similarly makes the unique application-to-serving handoff explicit:
  observed workflow state, Algorithm 1, and Request + Slack metadata.

## Iteration 0 - prior checked-in overview

Reviewer concerns:

- Five accent families made colors encode modules, algorithms, states, and
  arrows simultaneously.
- Blue, purple, and green all represented control-like traffic, so the legend
  could not answer whether an edge moved KV bytes.
- The detached failure strip read as a second diagram instead of a control path
  that drives the same data plane.
- Buses consumed space while hiding the direct source-target-Store transfer
  directions.

Decision: replace the palette and topology rather than restyle individual
boxes.

## Iteration 1 - two-color, three-plane redesign

Improvements:

- Reduced the visual vocabulary to neutral structure, blue control edges, and
  orange data edges.
- Put Algorithm 1 above the Router, Algorithms 2 and 3 inside the Controller,
  and Workers plus Store in one explicit data plane.
- Replaced the detached recovery strip with a Controller failure lane and
  direct KV movement.

Reviewer concerns after native Draw.io export:

- Fallback font selection made the first export look monospaced and forced
  plane titles onto two lines.
- The Algorithm 3-to-2 quota edge crossed the directory and later appeared to
  merge with the failure lane.
- The ACK/pressure feedback edge crossed Algorithm 3.
- Long orange labels obscured the modules they were meant to explain.

Decision: use the installed Noto Sans CJK SC family, route feedback along the
outer edge, number the three KV paths, and express Algorithm 2's alpha envelope
inside its own box.

## Final reviewer audit

- **Five-second story:** workflow state produces causal Slack; SlackLandlord
  independently selects `ADMIT_HBM`, `ADMIT_DRAM`, or `BYPASS` using
  pool-local victim frontiers; GuardedAIMD
  adjusts that envelope; Workers execute the commands and move KV directly.
- **Control versus data:** every arrow is either dark blue (`#1F4E79`) or
  orange (`#C65911`). The legend and plane titles repeat the distinction.
- **Physical path:** orange path 1 protects source KV in route-bound Peer HBM,
  path 2 protects it directly in Store DRAM, path 3 asynchronously establishes
  a DRAM backing from target HBM, and path 4 restores it to the recovery
  executor. KV does not enter the Controller.
- **Failure path:** fence, committed-state freeze, Slack-ordered
  restore/recompute, and output resume are one control-plane lane connected to
  the same Worker control bus.
- **Density and type:** the canvas uses three compact bands and 18-27 pt labels;
  there is no tiny explanatory footnote inside the figure.
- **Scope discipline:** the overview names SlackLandlord and GuardedAIMD but
  leaves heap internals, residency transitions, and formulae to the algorithm
  and patent figures. This prevents a main-paper overview from becoming a
  second pseudocode page.

The final source must be exported with the native Draw.io CLI and inspected
both standalone and on the compiled paper page before acceptance.
