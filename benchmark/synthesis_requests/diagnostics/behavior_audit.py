"""Static audit: SGLSim must not contain sim-only gates absent from SGLang."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

SCHEDULER = REPO / "python/sglsim/srt/managers/scheduler.py"
PREFILL_ADDER = REPO / "python/sglsim/srt/managers/prefill_adder.py"
UTILS = REPO / "benchmark/synthesis_requests/utils.py"
SRT_ROOT = REPO / "python/sglsim/srt"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def audit() -> int:
    failures: list[str] = []
    sched = _read(SCHEDULER)
    adder = _read(PREFILL_ADDER)
    utils = _read(UTILS)

    forbidden_in_scheduler = (
        ("_prefill_yields_to_decode", "sglang has no prefill-yield gate"),
        ("chunked_req_queue", "use single chunked_req like sglang"),
        ("token_usage >= 0.85", "sim-only KV usage waiting gate"),
        ("prefill_length <= chunk_cap", "sim-only short-chunk waiting filter"),
    )
    for needle, reason in forbidden_in_scheduler:
        if needle in sched:
            failures.append(f"FORBIDDEN '{needle}' in scheduler.py: {reason}")

    forbidden_in_utils = (
        ("0.10 * total_latency", "sim-only waiting cap for ref CSV"),
        ("inference = service_time", "sim-only inference rewrite masking scheduler tax"),
    )
    for needle, reason in forbidden_in_utils:
        if needle in utils:
            failures.append(f"FORBIDDEN '{needle}' in utils.py: {reason}")

    for py in SRT_ROOT.rglob("*.py"):
        text = _read(py)
        if "retract_cooldown" in text or "retract-cooldown" in text:
            failures.append(
                f"FORBIDDEN retract cooldown in {py.relative_to(REPO)}"
            )

    required_scheduler = (
        ("self.chunked_req", "single in-flight chunked prefill pointer"),
        ("for req in self.waiting_queue", "full waiting_queue walk"),
        ("check_hicache_events", "HiCache poll in scheduler"),
        ("def stash_chunked_request", "stash chunked prefill like sglang"),
        ("chunked_req_to_exclude", "exclude in-flight chunk from last_batch merge"),
        ("adder.new_chunked_req", "new_chunked_req handoff like sglang"),
        ("self._result_queue", "overlap result_queue pipeline"),
        ("def is_disable_overlap_for_batch", "consecutive-prefill overlap disable"),
        ("def _event_loop_step_overlap", "overlap event loop step"),
        ("def _advance_wall_clock_step", "wall-clock pipeline step sleep"),
        ("ForwardSlot", "forward slot CPU overlap"),
    )
    for needle, reason in required_scheduler:
        if needle not in sched:
            failures.append(f"MISSING '{needle}' in scheduler.py: {reason}")

    required_adder = (
        ("def _admit_kv_finalize", "load_back finalize before admission budgeting"),
        ("if loaded <= 0", "load_back failure must clear host_hit budget"),
        ("if self.cache.hierarchical_enabled", "host_hit/load_back only in HiCache mode"),
        ("def _new_request_decode_reserve", "new-req decode reserve without ratio"),
        ("input_tokens >= self.rem_input_tokens", "sglang >= boundary on rem_input"),
        ("// self.page_size * self.page_size", "page-aligned chunk truncation"),
    )
    for needle, reason in required_adder:
        if needle not in adder:
            failures.append(f"MISSING '{needle}' in prefill_adder.py: {reason}")

    print("behavior_audit")
    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("PASS: no forbidden sim-only patterns; required sglang mirrors present")
    return 0


def main() -> None:
    sys.exit(audit())


if __name__ == "__main__":
    main()
