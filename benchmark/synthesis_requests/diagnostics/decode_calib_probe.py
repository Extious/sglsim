"""Multi-point decode throughput calibration: sim predictor vs real fa3 logs."""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

from sglsim.srt.simulation.manager.config import ConfigManager  # noqa: E402
from sglsim.srt.time_predictor.base import ScheduleBatch as PredBatch  # noqa: E402
from sglsim.srt.time_predictor.base import ScheduleRequest  # noqa: E402

DEFAULT_CFG = REPO / "benchmark/synthesis_requests/config/server.json"
DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_REAL_LOG = REPO / "benchmark/synthesis_requests/results/without_hicache/server.log"

_DECODE_RE = re.compile(
    r"Decode batch, #running-req: (\d+), #token: (\d+).*"
    r"gen throughput \(token/s\): ([\d.]+)"
)


def _sim_tok_s(pred, bs: int, kv: int, decode_scale: float) -> float:
    pred.decode_scale_factor = decode_scale
    pred._static_latency_cache.clear()
    batch = PredBatch(
        reqs=[ScheduleRequest(extend_length=1, past_kv_length=kv) for _ in range(bs)]
    )
    step = pred.predict_infer_time(batch)
    return bs / step if step > 0 else 0.0


def _load_real_samples(log_path: Path) -> list[tuple[int, int, float]]:
    rows: list[tuple[int, int, float]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _DECODE_RE.search(line)
        if not match:
            continue
        bs, total, thr = int(match.group(1)), int(match.group(2)), float(match.group(3))
        if thr < 50:
            continue
        rows.append((bs, total // bs, thr))
    return rows


def main() -> None:
    cfg = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CFG
    model_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    real_log = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_REAL_LOG

    ConfigManager.load(str(cfg))
    pred = ConfigManager.get_inference_time_predictor(model_path)
    raw = json.loads(Path(cfg).read_text(encoding="utf-8"))
    decode_scale = float(raw.get("predictor", {}).get("decode_scale_factor", 1.0))

    print(f"config      = {cfg}")
    print(f"model       = {model_path}")
    print(f"real_log    = {real_log}")
    print(f"decode_scale= {decode_scale}\n")

    print("=== SIM predicted decode tok/s (scale=1.0) ===")
    print(f"{'bs':>4} {'kv':>8} {'tok/s':>10}")
    for bs in (4, 8, 10, 16):
        for kv in (8192, 16384, 24000):
            thr = _sim_tok_s(pred, bs, kv, decode_scale=1.0)
            print(f"{bs:>4} {kv:>8} {thr:>10.1f}")

    real = _load_real_samples(real_log)
    print("\n=== REAL fa3 decode tok/s (bs=10 KV bands) ===")
    ratios: list[float] = []
    for lo, hi in ((23000, 26000), (15000, 17500), (7500, 9000)):
        vals = [thr for bs, kv, thr in real if bs == 10 and lo <= kv <= hi]
        if not vals:
            print(f"  kv {lo}-{hi}: no samples")
            continue
        kv_mid = (lo + hi) // 2
        sim = _sim_tok_s(pred, 10, kv_mid, decode_scale=1.0)
        real_mean = statistics.mean(vals)
        ratio = real_mean / sim
        ratios.append(ratio)
        print(
            f"  kv {lo}-{hi}: n={len(vals)} real={real_mean:.1f} "
            f"sim={sim:.1f} ratio(real/sim)={ratio:.4f} "
            f"decode_scale~{1/ratio:.4f}"
        )

    if ratios:
        rec = statistics.mean(ratios)
        print(f"\nRecommended decode_scale_factor ≈ {1/rec:.3f} (mean real/sim ratio)")


if __name__ == "__main__":
    main()
