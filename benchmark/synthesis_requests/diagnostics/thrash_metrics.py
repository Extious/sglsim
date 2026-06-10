"""Extract retract-thrash indicators from server.log for sim vs sglang comparison."""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path


def analyze_server_log(path: Path) -> dict[str, float | int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    prefill_new_tok = [
        int(m.group(1))
        for m in re.finditer(
            r"Prefill batch, #new-seq: \d+, #new-token: (\d+)", text
        )
    ]
    cached_tok = [
        int(m.group(1))
        for m in re.finditer(r"#new-token: \d+, #cached-token: (\d+)", text)
    ]
    usage = [float(m.group(1)) for m in re.finditer(r"token usage: ([\d.]+)", text)]
    new_seq = [int(m.group(1)) for m in re.finditer(r"#new-seq: (\d+)", text)]
    running = [int(m.group(1)) for m in re.finditer(r"#running-req: (\d+)", text)]
    retract = len(re.findall(r"Retract requests", text, re.I))
    abort = len(re.findall(r"aborted last request", text, re.I))

    zero_cached = sum(1 for c in cached_tok if c == 0)
    n_prefill = len(prefill_new_tok)
    usage_sorted = sorted(usage) if usage else [0.0]

    return {
        "prefill_batches": n_prefill,
        "decode_batches": len(re.findall(r"Decode batch", text)),
        "retract_events": retract,
        "abort_events": abort,
        "prefill_new_token_sum": sum(prefill_new_tok),
        "cached_token_sum": sum(cached_tok),
        "zero_cached_prefill_batches": zero_cached,
        "zero_cached_fraction": zero_cached / n_prefill if n_prefill else 0.0,
        "token_usage_mean": statistics.mean(usage) if usage else 0.0,
        "token_usage_p90": usage_sorted[int(len(usage_sorted) * 0.9)]
        if usage_sorted
        else 0.0,
        "token_usage_max": max(usage) if usage else 0.0,
        "new_seq_mean": statistics.mean(new_seq) if new_seq else 0.0,
        "running_req_mean": statistics.mean(running) if running else 0.0,
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: thrash_metrics.py <server.log> [server.log ...]")
        sys.exit(1)
    for arg in sys.argv[1:]:
        path = Path(arg)
        m = analyze_server_log(path)
        print(f"\n=== {path} ===")
        for key, val in m.items():
            if isinstance(val, float):
                print(f"  {key}: {val:.3f}")
            else:
                print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
