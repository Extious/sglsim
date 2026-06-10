"""Probe the time predictor in isolation to separate code-logic from DB/calibration.

Builds the exact predictor used by a finished run (from its simulator_config.json)
and queries decode/prefill latency for representative (batch_size, isl) points, so we
can compare predicted per-step latency against sglang's measured per-step latency.
"""
import sys
from pathlib import Path

from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.time_predictor.base import ScheduleBatch as PredBatch
from sglsim.srt.time_predictor.base import ScheduleRequest


def decode_batch(batch_size: int, isl: int) -> PredBatch:
    return PredBatch(
        reqs=[ScheduleRequest(extend_length=1, past_kv_length=isl) for _ in range(batch_size)]
    )


def prefill_batch(extend: int, past: int) -> PredBatch:
    return PredBatch(reqs=[ScheduleRequest(extend_length=extend, past_kv_length=past)])


def main() -> None:
    cfg = sys.argv[1] if len(sys.argv) > 1 else "results/20260601_112755/simulator_config.json"
    ConfigManager.load(str(Path(cfg)))
    model_path = str((ConfigManager._raw or {}).get("model_path"))
    pred = ConfigManager.get_inference_time_predictor(model_path)
    print(f"predictor={pred.name} decode_scale={getattr(pred,'decode_scale_factor',None)} "
          f"prefill_scale={getattr(pred,'prefill_scale_factor',None)}")

    print("\n=== DECODE latency (ms/step) by (batch_size, isl) ===")
    for bs in (1, 2, 4, 6, 8):
        row = []
        for isl in (4096, 8192, 16384, 28000, 36000):
            ms = pred.predict_infer_time(decode_batch(bs, isl)) * 1e3
            row.append(f"isl{isl//1000}k={ms:7.1f}")
        print(f" bs={bs}: " + "  ".join(row))

    print("\n=== PREFILL latency (s) for extend=8192 over a cached prefix ===")
    for past in (0, 4096, 14336, 28000):
        s = pred.predict_infer_time(prefill_batch(8192, past))
        thr = 8192 / s if s > 0 else 0
        print(f" past={past:6d}: {s:6.3f}s  in_thru={thr:8.0f} tok/s")


if __name__ == "__main__":
    main()
