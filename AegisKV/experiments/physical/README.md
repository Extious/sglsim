# Physical calibration and matched replay

This directory is the fail-closed bridge from the completed metadata-DES to
future same-topology measurements. It does not submit jobs or execute failure
signals.

The three configuration layers are independent and explicit:

- `configs/request.json` binds the final audited simulation lineage and selects
  three seeds while retaining every non-seed axis, yielding 201 matched cells;
- `configs/failure.json` maps simulated fail-stop events to safe process-group
  `SIGKILL` injection and forbids GPU reset, reboot, power-cycle, or network
  partition;
- `configs/deployment.json` requires the exact 4-node x 8 A100-40GB topology,
  runtime/model identities, calibration points, transfer paths, and holdout
  error gate.

Create the immutable static plan:

```bash
uv run python AegisKV/experiments/physical/build_plan.py plan \
  --output AegisKV/experiments/physical/plans/a100_4x8_matched_replay_v1.json
```

Audit an inventory captured inside an exclusive allocation:

```bash
uv run python AegisKV/experiments/physical/build_plan.py preflight \
  --plan AegisKV/experiments/physical/plans/a100_4x8_matched_replay_v1.json \
  --inventory /path/to/inventory.json
```

A passing inventory authorizes calibration collection only. Matched replay,
performance results, and paper-ready claims remain false until a same-topology
profile covers Qwen3-8B/32B service time, all four source-owned transfer paths,
failure-detection/rebind delays, at least five independent repeats, and the
10% holdout-error gate.

`status/current_dp2_inventory.json` records the best historical local evidence:
one node with two A100-80GB GPUs. It is intentionally rejected by this contract
and is not a statement of current scheduler availability.
