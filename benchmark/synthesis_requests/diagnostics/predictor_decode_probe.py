"""Diagnostic probe for the aiconfigurator decode/prefill latency predictor.

Why this exists
---------------
SGLSim's simulated decode throughput (~181 tok/s for Qwen3-8B @ bs=10, ctx~24k)
was ~30% below the real SGLang trace (~255 tok/s). This script isolates the
root cause WITHOUT touching any scale factor, by querying the real predictor
directly and breaking the decode step down to the operator level.

Conclusion reached with this probe (see FINDINGS.md):
  * the decode step is ~80% generation_attention;
  * the aiconfigurator a100_sxm/sglang database only ships `flash_attention`
    kernel data, while the real trace ran `--attention-backend triton`;
  * the gap is a kernel mismatch (flash_attention vs triton), not a SGLSim bug,
    not an interpolation/extrapolation artifact, and not something to paper over
    with decode_scale_factor.

Usage
-----
    uv run python benchmark/synthesis_requests/diagnostics/predictor_decode_probe.py \
        [path/to/simulator_config.json] [model_path]

Defaults to the 20260531_142847 synthesis run config and Qwen/Qwen3-8B.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

from sglsim.srt.simulation.manager.config import ConfigManager  # noqa: E402
from sglsim.srt.time_predictor.base import ScheduleBatch as PredBatch  # noqa: E402
from sglsim.srt.time_predictor.base import ScheduleRequest  # noqa: E402

DEFAULT_CFG = (
    REPO
    / "benchmark/synthesis_requests/results/20260531_142847/simulator_config.json"
)
DEFAULT_MODEL = "Qwen/Qwen3-8B"

# Qwen3-8B / A100-SXM constants used for the SOL (memory-bound) lower bound.
NUM_LAYERS = 36
N_Q_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
DTYPE_BYTES = 2  # bf16
A100_SXM_HBM_BW = 2039e9  # bytes/s


def _decode_batch(bs: int, past_kv: int) -> PredBatch:
    return PredBatch(
        reqs=[ScheduleRequest(extend_length=1, past_kv_length=past_kv) for _ in range(bs)]
    )


def _prefill_batch(bs: int, extend: int, past: int) -> PredBatch:
    return PredBatch(
        reqs=[ScheduleRequest(extend_length=extend, past_kv_length=past) for _ in range(bs)]
    )


def main() -> None:
    cfg = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CFG
    model_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL

    ConfigManager.load(str(cfg))
    pred = ConfigManager.get_inference_time_predictor(model_path)
    print(f"config    = {cfg}")
    print(f"model     = {model_path}")
    print(f"predictor = {type(pred).__name__}\n")

    # --- 1. decode throughput vs context (interpolation is smooth, not stepped) ---
    print("=== DECODE step latency vs past_kv (bs=10) ===")
    print(f"{'past_kv':>8} {'step_ms':>10} {'tok/s(10)':>10}")
    for past in [8192, 16384, 20000, 24000, 25000, 30000, 40000]:
        pred._static_latency_cache.clear()
        t = pred.predict_infer_time(_decode_batch(10, past))
        print(f"{past:>8} {t * 1e3:>10.3f} {(10.0 / t if t > 0 else 0):>10.1f}")

    # --- 2. decode throughput vs batch size ---
    print("\n=== DECODE step latency vs batch_size (past_kv=24000) ===")
    for bs in [1, 4, 8, 10, 16]:
        pred._static_latency_cache.clear()
        t = pred.predict_infer_time(_decode_batch(bs, 24000))
        flag = "  <-- OOM (negative)" if t < 0 else ""
        print(f"bs={bs:>3}: step={t * 1e3:>9.3f}ms tok/s={(bs / t if t > 0 else 0):>7.1f}{flag}")

    # --- 3. operator breakdown of one decode step ---
    print("\n=== DECODE operator breakdown (bs=10 isl=24000) ===")
    sess = pred._session
    rc = pred._runtime_config_cls(batch_size=10, isl=24000, osl=2)
    summary = sess.run_static(rc, mode="static_gen")
    d = summary.get_generation_latency_dict()
    tot = sum(d.values())
    print(f"  total step = {tot:.3f} ms  -> {10 / (tot / 1e3):.1f} tok/s")
    for k, v in sorted(d.items(), key=lambda x: -x[1])[:6]:
        print(f"    {k:<28} {v:8.4f} ms  ({100 * v / tot:4.1f}%)")

    # --- 4. database-mode and SOL comparison ---
    print("\n=== database_mode comparison (bs=10 isl=24000, decode step) ===")
    from aiconfigurator.sdk.common import DatabaseMode

    db = sess._database
    for mode in ["SILICON", "EMPIRICAL", "SOL"]:
        try:
            db.set_default_database_mode(getattr(DatabaseMode, mode))
            s = sess.run_static(
                pred._runtime_config_cls(batch_size=10, isl=24000, osl=2),
                mode="static_gen",
            )
            step = sum(s.get_generation_latency_dict().values())
            print(f"  {mode:<10}: step={step:8.3f}ms  tok/s={10 / (step / 1e3):.1f}")
        except Exception as e:  # noqa: BLE001
            print(f"  {mode:<10}: ERR {type(e).__name__}: {str(e)[:70]}")
    db.set_default_database_mode(DatabaseMode.SILICON)

    # SOL memory-bound lower bound for one decode step (KV read only).
    kv_bytes = 10 * 24000 * (2 * N_KV_HEADS * HEAD_DIM * DTYPE_BYTES * NUM_LAYERS)
    print(
        f"  SOL(mem KV-read): {kv_bytes / A100_SXM_HBM_BW * 1e3:.2f} ms "
        f"({kv_bytes / 1e9:.1f} GB @ {A100_SXM_HBM_BW / 1e9:.0f} GB/s)"
    )

    # --- 5. measured-range coverage of generation_attention data ---
    print("\n=== generation_attention measured-range coverage ===")
    print("  NOTE: raw a100_sxm/sglang data measures decode step only up to 8191;")
    print("        ctx~24000 is LINEAR extrapolation (smooth, no jump -> not the cause).")
    print("        Only kernel_source=flash_attention exists; real trace used triton.")


if __name__ == "__main__":
    main()
