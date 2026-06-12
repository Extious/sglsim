from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = Path("/home/comp/zhanzhao/sglang/python/sglang")

PARITY_FILES = [
    "srt/entrypoints/http_server.py",
    "srt/entrypoints/engine.py",
    "srt/managers/tokenizer_manager.py",
    "srt/managers/data_parallel_controller.py",
    "srt/managers/scheduler.py",
    "srt/managers/schedule_batch.py",
    "srt/managers/schedule_policy.py",
    "srt/managers/scheduler_output_processor_mixin.py",
    "srt/managers/scheduler_runtime_checker_mixin.py",
    "srt/managers/cache_controller.py",
    "srt/mem_cache/radix_cache.py",
    "srt/mem_cache/hiradix_cache.py",
    "srt/mem_cache/hicache_storage.py",
    "srt/mem_cache/memory_pool.py",
    "srt/mem_cache/memory_pool_host.py",
    "srt/mem_cache/allocator.py",
    "srt/model_executor/model_runner.py",
    "srt/server_args.py",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest() -> dict[str, dict[str, str]]:
    data: dict[str, dict[str, str]] = {}
    for rel in PARITY_FILES:
        upstream = UPSTREAM / rel
        local = REPO / "python" / "sglsim" / rel
        data[rel] = {
            "upstream_exists": str(upstream.is_file()),
            "local_exists": str(local.is_file()),
            "upstream_sha256": _sha256(upstream) if upstream.is_file() else "",
            "local_sha256": _sha256(local) if local.is_file() else "",
        }
    return data


def main() -> int:
    print(json.dumps(build_manifest(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
