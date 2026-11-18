# Physical calibration status

Observed on 2026-08-29 (Asia/Hong_Kong): the local `short` partition exposes
`gpu25`--`gpu28`, each with two Slurm `gpu:a100` resources. All eight resources
were allocated at the earlier inspection. A later read-only refresh after the
`20260829T051356Z_paper_complete_v2` run showed both GPUs on `gpu27` unallocated,
while `gpu25`, `gpu26`, and `gpu28` still reported allocated A100 resources.
This is scheduler availability only, not authorization or a topology match.
Slurm's auxiliary `gpu_mem:32K` label is not used as the hardware authority.

The most recent clean, exclusive-node inventory is:

```text
0, NVIDIA A100 80GB PCIe, 81920 MiB
1, NVIDIA A100 80GB PCIe, 81920 MiB
```

Its audit is the external, read-only artifact
`/home/comp/zhanzhao/aegiskv-execution/qwen3-a100-dp2-20260824-v1-v19-r3/aegiskv/runs/a100-dp2-clean-preflight-135865-20260824T081223Z/clean-gpu-preflight-audit.json`.
That audit passes clean-resource and topology preflight only and explicitly is
not model execution, transfer calibration, performance, or paper-ready evidence.

The current S1--S9 plus E1 deployment is intentionally fixed at four nodes, eight
NVIDIA A100 40GB GPUs per node. Therefore the available 1-node 2xA100-80GB
machine is not a same-topology calibration source. A user-held Slurm job
`135889` (`aegis-a100-model`) targets the separate 1-node DP2 campaign and was
still `PENDING (JobHeldUser)` at the later refresh; this experiment workflow did
not release, rebind, or resubmit it.

Consequences:

- S1--S9 and E1 remain complete metadata-DES experiments with modeled service time;
- A100-80GB DP2 measurements may be collected later as diagnostic evidence,
  but cannot silently replace A100-40GB 32-GPU coefficients;
- clearing `paper_ready_calibrated=false` requires either access to the modeled
  4x8 A100-40GB topology, or an explicit deployment redesign followed by a new
  simulation lineage and matched physical replay.

The current executable handoff is under `../physical/`. Its three JSON layers
bind the final `20260829T064719Z_paper_complete_v2` lineage and materialize a
201-cell plan: the three numerically smallest preregistered seeds per group,
with every non-seed S1--S9/E1 axis retained. The plan hashes the source
lineage, all selected raw/shadow artifacts, physical configs, and plan-builder
code; it contains no simulated WCT or benefit outcomes.

`../physical/status/current_dp2_preflight.json` applies that contract to the
historical DP2 inventory and returns `BLOCKED_TOPOLOGY_OR_IDENTITY`. This is a
deliberate negative preflight, not a failed physical experiment. No job was
released, rebound, submitted, or executed while creating it.
