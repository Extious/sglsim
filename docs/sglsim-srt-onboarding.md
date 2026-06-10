# SGLSim SRT Onboarding

SGLSim mirrors SGLang Runtime (`sglang.srt`) under `sglsim.srt`.

| SGLang | SGLSim |
|--------|--------|
| `sglang.launch_server` | `python -m sglsim.launch_server` |
| `sglang.srt.managers.tokenizer_manager` | `sglsim.srt.managers.tokenizer_manager` |
| `sglang.srt.managers.scheduler` | `sglsim.srt.managers.scheduler` |
| `sglang.srt.managers.detokenizer_manager` | `sglsim.srt.managers.detokenizer_manager` |
| `sglang.srt.managers.cache_controller` | `sglsim.srt.managers.cache_controller` |
| `sglang.srt.mem_cache.hiradix_cache` | `sglsim.srt.mem_cache.hiradix_cache` |
| `sglang.srt.entrypoints.engine` | `sglsim.srt.entrypoints.engine` |

ZMQ is simulated via `sglsim.srt.utils.network.SimZmqSocket` (in-memory queues).

Modes: `--streaming-arrival` (required for HTTP server). Main uses wall-clock sleep only. Future logical-clock mode: [offline-simulate-design.md](todo/offline-simulate-design.md).

Scope: [sglsim-srt-contract.md](sglsim-srt-contract.md).
