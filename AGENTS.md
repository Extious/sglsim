# SGLSim Agent 项目说明

SGLSim 是 SGLang 服务的 **纯元数据、在线模拟器**：启动 **真实 SGLang 进程拓扑**（`Engine` + `DataParallelController` + 每个 `dp_rank` 一个 `Scheduler` + `DetokenizerManager`），背后挂 **实时 HTTP 服务**，由 **真实异步闭环客户端** 驱动——但将 GPU forward 替换为预测时延（无 CUDA/NCCL）。

运行时代码 **仅** 使用 `python/sglsim/srt/`。旧目录 `core/`、`scheduler/`、`kvcache/`、`runtime/`、`comm/` 已移除。

## SGLang 参考（语义来源）

Pinned SGLang 是所有运行时语义的 **权威参照**。不要猜测行为——改代码前先读对应 SGLang 源码。

| | 路径 |
|---|---|
| SGLang 仓库 | `/home/comp/zhanzhao/sglang` |
| SGLang SRT 根目录 | `/home/comp/zhanzhao/sglang/python/sglang/srt/` |
| SGLSim SRT 根目录 | `python/sglsim/srt/` |

- 模块对照：`docs/sglsim-srt-onboarding.md`
- 对齐记录：`benchmark/synthesis_requests/diagnostics/alignment_notes.md`、`reference/sglang_notes.md`
- 保真度 / trace 差异：`benchmark/synthesis_requests/diagnostics/FINDINGS.md`
- 运行时 **禁止** `import sglang`；在 `sglsim.srt.*` 下镜像逻辑

## 开发要求

**严格对齐 SGLang。** 所有代码——调度、batch 生命周期、radix/HiCache、DP/IPC 拓扑、指标——必须 **严格对齐** 上述 pinned SGLang。修改行为时，应标注对应 SGLang 文件/函数，并保持控制流等价。

**唯一替换点。** 唯一有意替换的是 GPU 推理路径：真实 CUDA/Triton/NCCL forward → **AIConfigurator** 延迟预测（`python/sglsim/srt/time_predictor/` 中的 `InferTimePredictor`）。每 event-loop step **一次** `sleep(forward + cpu_overflow)`（overlap ON，见下）；元数据副作用（KV 分配、cache 事件、队列状态迁移）仍按 SGLang 顺序执行。

**允许的 sim-only 层**（不得改变调度语义）：

- `SimZmqSocket` / `SimIpcRegistry`（进程内队列替代 ZMQ）
- `MockModel` / 元数据 KV pool（无 torch forward）
- AIConfigurator / 启发式时延预测与通信时延预测器

**避免** 偏离 SGLang 的 sim-only 捷径（如额外 event-loop 门控、不同 admission 规则）。回归守卫：`benchmark/synthesis_requests/diagnostics/behavior_audit.py`、`test/unit/srt/test_prefill_adder_sglang_parity.py`。

## 时间模型（墙钟）

真实墙钟 + **Pipeline v2 真机式 overlap**（`python/sglsim/srt/simulation/`）：

- `run_batch` 仅 predict，写 `batch.inference_dur_s`，**不 sleep**。
- 每步 `_advance_wall_clock_step`：`compute_pipeline_sleep_s` → overlap ON 时 `sleep(forward + cpu_overflow)`；overlap OFF（连续 prefill）时 serial `h2d + forward + backup`。
- H2D/D2H 经 `check_hicache_events` / `consume_backup_*` 簿记，**不占独立 sleep**（对齐真机 stream overlap）。
- `compute_visible_gpu` 保留为诊断（`comm_h2d_visible_s` / `comm_d2h_remainder_s`），不驱动 overlap ON 的墙钟。
- `ForwardSlot` 用上一拍 `slot_end_wall` 给出 CPU budget；调度耗时超出 budget → `cpu_overflow` 并入 step sleep。
- 各子进程在同一真实时间线上独立 sleep，多 DP 漂移自然产生。

离线逻辑时钟不在 main 实现；设计见 `docs/todo/offline-simulate-design.md`。main **禁止**保留逻辑时钟分支或 `SimGlobalClock` API。

## 模拟器 landscape

Vidur / Opal-sim / sglang-simulator 是单进程、离线、逻辑时间 DES（trace replay）。SGLSim 的差异点是 **多真实子进程 + 实时 HTTP 服务 + 真实 async 客户端（在线）**。

## 验证

```bash
uv run --extra test pytest test/unit test/integration -q
```

## 分支

- `main`：在线 serving 模拟器（parity / HiCache / Pipeline v2）；发布 tag `v0.1.0` 起不含 fault tolerance。
- `fault-tolerance`：在 `main` 基础上增加故障注入、离线策略模拟、multi-agent benchmark 等（见该分支 README）。

## 在线 benchmark

- **Synthesis**（`benchmark/synthesis_requests/run.py`）：HTTP 服务 + `--streaming-arrival`、`SimulationMode.STREAMING`；client worker 调用 `POST /v1/completions`，由 `max_concurrency` 限流（`request_rate=inf`）。
- Pipeline v2 step sleep（overlap ON 仅 forward + CPU overflow；传输簿记不占墙钟）。
- 双目标诊断脚本：`benchmark/synthesis_requests/diagnostics/baseline_report.py`、`scale_matrix.py`（用于 forward/makespan 权衡取证与 run 质量检查）。

## Multi-DP（`dp_size > 1`）

- 父进程：`Engine` + `DataParallelController`（round-robin）+ `DetokenizerManager`；子进程：每个 `dp_rank` 一个 `run_scheduler_process`。
- IPC：`python/sglsim/srt/utils/network.py`（`SimIpcRegistry`、`build_dp_ipc_registry`）。

## 调度（continuous batching）

- 默认 scheduler 镜像 SGLang：持久 `running_batch`，每步 prefill-first（`get_next_batch_to_run`），prefill 后 merge，decode 走 `update_running_batch`。Prefill 准入由 `PrefillAdder` 控制（`new_token_ratio`、`batch_is_full`、chunked prefill）。
- 每个 decode step 在 `update_running_batch`（`run_batch` 之前）分配 1 个 KV token，镜像 SGLang `prepare_for_decode` → `alloc_for_decode`；`process_batch_result` 仅追加输出 token。
- 与真实 trace 的保真度说明：`benchmark/synthesis_requests/diagnostics/FINDINGS.md`（fa3 baseline、kernel 对齐）。

## HiCache

- L1/L2/L3 元数据 + 预测通信时延：`python/sglsim/srt/mem_cache/`（`HiRadixCache`、`HiCacheController`、`MockHiCacheStorage`）。
- Write policy（`write_through` / `write_through_selective` 默认 / `write_back`）、`load_back_threshold`、L3 prefetch/backup；D2H overlap 在 `_response_time_after_batch` 经 `consume_backup_overlap(forward)` 簿记（advance 仅 sleep forward）。

## 关键路径

- SGLang SRT（参照）：`/home/comp/zhanzhao/sglang/python/sglang/srt/`
- SGLSim 运行时：`python/sglsim/srt/`
- CLI：`python/sglsim/launch_server.py`
- 时延预测器：`python/sglsim/srt/time_predictor/`
- 文档索引：`docs/README.md`
- Synthesis workload：`benchmark/synthesis_requests/`；参考笔记：`reference/`
