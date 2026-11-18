# System overview: ten-round visual iteration log

Date: 2026-08-03 Asia/Hong_Kong.

Reference style:
`/home/comp/zhanzhao/sglang/benchmark/aegiskv_mooncake/docs/mooncake-hicache-control-data-plane.drawio`.

Each round regenerated the native Draw.io XML, ran the structural validator,
rendered a new PNG, and was visually inspected before the next edit. The
validator reports vertex overlap, container overflow, missing cells, invalid
edge geometry, evidence failures, and style/content-contract failures
separately, so a non-overlap result is not inferred from a generic exit code.

## Iteration 1: explicit algorithm products

Preview: `iteration_01_algorithm_outputs.png`

- Added explicit output tickets for Algorithm 1, 2, and 3.
- Added the commit rule next to the three algorithm products.
- Routed A1 scope/Slack, A2 plan, and A3 budget toward their consumers.
- Visual inspection: no node overlap; output pills were too long and some text
  was visually abbreviated, which became the target of iteration 2.

## Iteration 2: input, decision, and check semantics

Preview: `iteration_02_algorithm_io.png`

- Rewrote every algorithm box as INPUT / DECIDE / CHECK where applicable.
- Shortened output tickets and restored direct A1/A3-to-A2 control links.
- Visual inspection: control flow became readable and no nodes overlapped.
- The content gate found that searchable `alpha` and `beta` words had been
  reduced to Greek symbols only; geometry/overlap checks had no issue. This
  content regression was repaired in iteration 3.

## Iteration 3: plan to execution to commit

Preview: `iteration_03_plan_execute_commit.png`

- Restored `alpha (α)` and `beta (β)` in the A3 budget.
- Increased the control-plane output area.
- Added a central A2 execution gate between plan output and group commit.
- Visual inspection: no node overlap; the new execution gate was readable but
  edge semantics were not yet visible in the preview renderer.

## Iteration 4: visible arrows and edge labels

Preview: `iteration_04_edge_labels.png`

- Extended the renderer with arrowheads and edge labels.
- Widened the A2 execution gate.
- Visual inspection exposed previously hidden clutter: the long A1-to-RTO line
  crossed the data plane, D2H labels were duplicated, and recovery labels were
  too long. No vertex overlap was present.

## Iteration 5: routed feedback and reduced line noise

Preview: `iteration_05_routed_feedback.png`

- Removed the redundant long A1-to-RTO connector; the RTO node now carries
  the A1 input explicitly.
- Created separate corridors for A3 budget publication and sealed-window
  telemetry feedback.
- Shortened repeated path labels and retained only action-relevant labels.
- Visual inspection: the central data plane became substantially cleaner; no
  node overlap or container overflow was reported.

## Iteration 6: unified Algorithm 2 execution vocabulary

Preview: `iteration_06_a2_failure_path.png`

- Labeled the A2 stages directly: LOCAL, REMOTE, L3, COMMIT, CHECK, RECOVER,
  and FAILURE BROKER.
- Rewrote the recovery-band header as an ordered A2 failure path.
- Removed the RTO label that sat directly on a card boundary.
- The first pass exposed a case-sensitive `Failure` content-token regression,
  not a geometry problem; it was corrected and the round was rerendered.
- Final visual inspection and strict validation passed with no overlap.

## Iteration 7: action-oriented closed loop

Preview: `iteration_07_action_labels.png`

- Standardized inter-algorithm verbs: CONSUME, EXECUTE, COMMIT, PUBLISH, SEAL.
- Made the control-plane header summarize A1 ANALYZE, A2 PLACE/RECOVER, and A3
  ADAPT.
- Enlarged and simplified the central commit node.
- Replaced unsupported decorative numbering with plain numeric steps.
- Visual inspection: algorithm flow was readable in both directions and no
  geometry overlap was present.

## Iteration 8: evidence and A3 readability

Preview: `iteration_08_evidence_readability.png`

- Made A3 consume sealed-window telemetry explicitly.
- Renamed the active profiler box as the A3 feedback source.
- Expanded the profiler calibration strip to two complete lines.
- Visual inspection: measured evidence no longer truncated visually; strict
  overlap and evidence validation passed.

## Iteration 9: colored path legend and clipping audit

Preview: `iteration_09_colored_legend.png`

- Replaced the small prose legend with four color-coded path chips: control,
  local, remote scope-durable, and persistent L3.
- Added renderer-level reporting for text clipped by the preview layout.
- Visual inspection: layout remained balanced and overlap-free.
- The new audit reported two clipped nodes (`commit-rule` and
  `active-profiler`), which became the only remaining final-round fixes.

## Iteration 10: zero-clipping final

Preview: `iteration_10_final.png`

- Increased the two reported node heights without moving neighboring nodes.
- Extended the strict validator to require the algorithm output tickets,
  execution gate, action labels, A2 stages, A3 feedback, and four-part legend.
- Final renderer result: `clipped_cells=none`.
- Final strict result: 57 cells, 35 vertices, 20 edges, accepted Slurm job
  67111, and no issues.
- Final visual inspection: no overlapping cards, no truncated card text, clear
  control/data/recovery bands, and a complete A1 to A2 to A3 feedback loop.

## Preview directory

All round artifacts are under:

`/home/comp/zhanzhao/sglsim/tmp/pdfs/system_overview_pp2_story/`

The directory contains the baseline plus `iteration_01` through
`iteration_10`; all eleven PNG files are non-empty and were rendered from the
corresponding round's generated XML.

## Post-review overlap correction

Preview: `system_overview_mooncake_style_67111.png`

The first ten-round validator only proved that sibling cards did not overlap.
It did not detect floating edge-label boxes, connectors crossing foreground
cards, connector-to-connector intersections, or long routes that visually
merged with container borders. A user review correctly exposed those gaps.

- Moved CONSUME, EXECUTE, COMMIT, PUBLISH, and recovery semantics into the
  corresponding nodes and removed every floating edge label.
- Deleted redundant long feedback, frontier, commit, and replacement-compute
  routes; the remaining 14 connectors now express only local adjacency.
- Re-routed PP-to-Host, A2 plan-to-execution, and commit-to-L3 paths through
  dedicated whitespace corridors.
- Extended strict validation with edge-label, edge-to-node, and edge-to-edge
  collision gates.
- Final strict result: 51 cells, 35 vertices, 14 edges, 0 edge labels, 0
  edge-to-node collisions, 0 edge-to-edge intersections, accepted Slurm job
  67111, and no issues.
- Final renderer result: `clipped_cells=none`; visual inspection confirms that
  no connector is drawn through a card, label, or sibling connector.
