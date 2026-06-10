# Mini-SGLang Reference Notes

Naming aligned with Mini-SGLang; implementation under `sglsim.srt`:

- `GenerateReqInput` / schedule batch (Req lifecycle)
- `Scheduler` in `sglsim.srt.managers.scheduler`
- `Engine` in `sglsim.srt.entrypoints.engine`

SGLSim adds metadata Mini-SGLang does not model:

- DP/TP/PP topology (`sglsim.srt.distributed`)
- HiCache L1/L2/L3 (`sglsim.srt.mem_cache`, `managers.cache_controller`)
- transfer timing and benchmark metrics

Metadata-only: no model tensor execution.
