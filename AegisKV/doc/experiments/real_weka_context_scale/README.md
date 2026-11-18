# Real-Weka context projection gate

This directory prevents a replacement V100 profile from silently promoting the
old unscaled Qwen3-8B replay. The pinned Weka trace contains 98,827 requests;
85,129 (86.14%) exceed Qwen3-8B's 40,960-token executable context, and 95,348
(96.48%) exceed the current 4,096-token physical calibration range.

The context-only audit preregisters every scale before any replacement
performance run:

| Scale | Maximum context | Gate | Role |
|---:|---:|---|---|
| 1.000 | 996,579 | invalid for Qwen3-8B | diagnostic only |
| 0.004 | 3,987 | within model and calibration range | primary candidate after the V100 schema-v2 gate |
| 0.008 | 7,974 | model-valid, calibration extrapolation | sensitivity |
| 0.016 | 15,947 | model-valid, calibration extrapolation | sensitivity |
| 0.032 | 31,892 | model-valid, calibration extrapolation | sensitivity |

All scale outcomes must be retained. The primary scale cannot be changed after
observing treatment results. Scaling preserves workflow topology but changes KV
volume, so no scaled result may be described as an unmodified Weka replay.

Run the immutable audit from the repository root:

```bash
uv run python \
  AegisKV/experiments/real_weka_context_scale/audit_context_scale.py \
  --config AegisKV/experiments/real_weka_context_scale/config.json \
  --output /path/to/new-context-audit.json
```

The checked output is a feasibility/preregistration artifact, not a simulation
performance result and not physical or paper-ready evidence.
