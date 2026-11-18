# fig_system_overview_generated_v2_tight.png Prompt

Create a publication-ready system overview diagram for an academic systems-paper system overview figure.

Output target: `fig_system_overview_generated_v2_tight.png`.

Visual constraints:
- pure white background with one unified rounded canvas
- landscape slide aspect
- readable at beamer slide size
- no logos
- no watermark
- no decorative gradients
- minimize crossing arrows and avoid label overlap

The rendered asset should present AegisKV as one continuous serving loop rather than isolated modules. It should use three connected lanes:

1. Agentic workflow: a 1-3-1-3-1 workflow with Critical and Non-Critical fan-out agents and visible failure candidates.
2. Control plane: request metadata, Criticality classifier, MDP policy plus online alpha, peer-rank selector, and retry plan.
3. Data plane: SGLSim online server, DP ranks with inference/backup HBM partitions, Host backup directory, and peer/host/recompute replay paths.

The figure should make AegisKV visually distinct from One Pool, Two Caches by emphasizing workflow-level failure scope, criticality-aware alpha control, and peer GPU / host / recompute recovery paths rather than EMB/KV cache partition tables.
