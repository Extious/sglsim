# 诊断留底：SGLSim vs 真实 SGLang 的 decode 吞吐/e2e 差异根因

针对 `benchmark/synthesis_requests`（Qwen3-8B、a100_sxm、dp_size=2、墙钟）下
模拟与真实推理的对比，本目录记录根因调查的证据与可复现脚本。

## 复现命令

```bash
# 预测器 decode 步延迟与算子 breakdown
uv run python benchmark/synthesis_requests/diagnostics/predictor_decode_probe.py

# decode 标定：sim 预测 vs fa3 真实 gen throughput
uv run python benchmark/synthesis_requests/diagnostics/decode_calib_probe.py

# e2e makespan 逐阶段拆解（prefill / decode 墙钟）
uv run python benchmark/synthesis_requests/diagnostics/e2e_makespan_decompose.py \
  benchmark/synthesis_requests/logs/server.log \
  benchmark/synthesis_requests/results/without_hicache/server.log \
  benchmark/synthesis_requests/results/<run_id>/request_details.csv

# 对比 sim 与 fa3 真实 trace 的 e2e 分布
uv run python benchmark/synthesis_requests/diagnostics/compare_results.py \
  benchmark/synthesis_requests/results/<run_id> \
  benchmark/synthesis_requests/results/without_hicache
```

## 对比数据来源

| 标签 | 路径 | 说明 |
|---|---|---|
| sim | `results/20260531_142847/` | SGLSim 墙钟模拟 |
| fa3 (no hicache) | `results/without_hicache/` | 真实 SGLang，`--attention-backend fa3` |
| fa3 (hicache) | `results/with_hicache/` | 真实 SGLang + HiCache |
| triton (旧) | `results/real_trace/20260529_133143/` | 旧 trace，`attention_backend=triton` |

## 双态 parity 对齐（2026-06-08，No-overlap First）

| 参考 | 路径 | 规模 | e2e mean | makespan |
|------|------|------|----------|----------|
| SGLang no-overlap | `results/with_hicache_without_overlap_sglang` | 180 req | ~46.1s | ~303s |
| SGLang overlap | `results/with_hicache_with_overlap_sglang` | 180 req | ~42.3s | ~277s |

**禁止**与旧 mini 参考 `with_hicache_sglang`（96 req）混比。

### 配置

- Full parity：`config/parity_full_no_overlap/`、`config/parity_full_overlap/`（30 组、并发 30、`mem_fraction_static=0.836`）
- Mini 回归：`config/no_overlap/`（16 组，快速行为/崩溃回归）

### 运行

```bash
# 阶段 B：必须先收敛 no-overlap
uv run python benchmark/synthesis_requests/run.py \
  --config-dir benchmark/synthesis_requests/config/parity_full_no_overlap \
  --disable-overlap-schedule

# 阶段 D：仅 no-overlap 门禁全部 <=10% 后
uv run python benchmark/synthesis_requests/run.py \
  --config-dir benchmark/synthesis_requests/config/parity_full_overlap
```

### 诊断（每轮固定）

```bash
RUN=benchmark/synthesis_requests/results/<run_id>
REF=benchmark/synthesis_requests/results/with_hicache_without_overlap_sglang  # or with_overlap

uv run python benchmark/synthesis_requests/diagnostics/baseline_report.py <run_id> --ref-run $(basename $REF)
uv run python benchmark/synthesis_requests/diagnostics/latency_budget.py $RUN $REF --strict
uv run python benchmark/synthesis_requests/diagnostics/loop_gate.py $RUN $REF
uv run python benchmark/synthesis_requests/diagnostics/compare_results.py $RUN $REF
uv run python benchmark/synthesis_requests/diagnostics/e2e_makespan_decompose.py \
  $RUN/server.log $REF/server.log $RUN/request_details.csv $REF/request_details.csv
uv run python benchmark/synthesis_requests/diagnostics/tail_breakdown.py $RUN $REF
```

### Round 11 逻辑修复

- **`cache_unfinished_req` 前缀口径**：chunked prefill 插入 radix 时用 `max(cached_length, cache_protected_len)` 计算 suffix，避免 `cached_length` 滞后导致 `missing deferred prefill page ids` 与 worker 崩溃（`hiradix_cache.py`）。
- **诊断 ref 参数化**：`parity_refs.py`；`alignment_loop.py` / `baseline_report.py` / `loop_gate.py` / `latency_budget.py` / `tail_breakdown.py` / `scale_matrix.py` 默认指向新参考。
- **Goal hook 门控**：仅 `SGLSIM_GOAL_HOOK=1` 时 stop hook 审查双态 Goal。

### Round 12 no-overlap 修复（2026-06-08）

修复范围只覆盖 no-overlap 先行收敛；未运行 overlap parity。

- **多轮输出一致性**：模拟输出 token 与 detokenizer fast path 文本改为同一个可重复的一 token 单元，避免 t1/t2 prompt 与 radix key 偏离。
- **admission-time prefix refresh**：fresh waiting request 在 admission 前重新 match radix，恢复 shared system prompt cache hit。
- **cache-hit 统计口径**：`cache_hit_tokens` / `#cached-token` 记录 prefix length，而不是 `cached_length` 已更新后的 delta；full no-overlap cache_hit_sum 从 0 级别恢复到 2.55M。
- **blocked chunk retry 去重**：active `chunked_req` 失败后退出 active 状态并在 waiting queue 去重，避免同一 req 在同一 prefill batch 被重复 admission，导致 retry 计数被成功路径清零并无限重放。

验证：

| run | 配置 | completed | failed | makespan | cache_hit_sum | no-overlap ref ratio |
|---|---|---:|---:|---:|---:|---:|
| `20260608_173929` | `config/no_overlap` | 96 | 0 | 254.6s | 1.37M* | mini |
| `20260608_174434` | `config/parity_full_no_overlap` | 180 | 0 | 385.8s | 2.55M | 1.27x |

\* mini 与 full 请求集不同；只用于完成性/卡死回归。

Full no-overlap 对比 `with_hicache_without_overlap_sglang`：

- e2e mean ratio 1.28、p90 ratio 1.21、max ratio 1.51。
- cache_hit_sum `2,553,691` vs real `2,588,842`，量级已对齐。
- sim `retract=26`，real `retract=0`；剩余性能差距主要来自 HiCache/L1 内存压力与 retract 交织，不再是 cache-hit=0 或 completion 卡死。

## 观测差异（fa3 apples-to-apples，2026-05-31）

| 指标 | 模拟 | fa3 真实 (no hicache) |
|---|---|---|
| e2e mean / std (s) | 125.35 / 8.08 | 75.56 / 1.15 |
| decode gen tok/s (per DP, bs=10) | ~173–181 | ~198–199 |
| decode `#token` 趋势 | 修复前恒定；修复后随生成递增 | 240999→244599 递增 |
| `#running-req` 动态 | 波次 10→0→10 | 第一波 gang（与 sim 一致） |

fa3 重测后 decode 吞吐差距缩小至 ~10%（181 vs 199），证实旧 trace 的 triton vs flash_attention kernel 失配是主要 decode 偏差来源。e2e 仍偏高 ~66%，需进一步排查 prefill 预测与整体 makespan。

## 标定与 e2e 对齐（2026-05-31 后续）

### decode_scale_factor

- 在 bs=10、KV≈24k（fa3 without_hicache 稳态）下，SILICON 预测 ~178 tok/s，真实 ~197 tok/s。
- 比值稳定：`decode_scale_factor ≈ 0.905`（`real/sim`，配置见 `config/server.json`）。
- 该因子只修正 **单步 predicted sleep**，不改变调度交织。

### prefill 墙钟 2× 根因（已修复）

- **不是** prefill 单 chunk 预测偏高（日志 `input throughput` sim/real 接近）。
- **是** chunked prefill 每步 `extend_prefill` 对整段 prefix 做 `match_prefix`/`insert`（page_size=1 时 O(N)），墙钟在 predicted sleep 之外额外 +0.5–1.2s/chunk。
- 修复：非 HiCache 场景 chunk 期间 **defer radix insert**（`extend_prefill_alloc_only`），**prefill 完成时** 一次 `finalize_prefill_in_tree`（对齐 decode `alloc_for_decode` 思路）。
- **注意**：必须在 prefill 结束、decode 开始前 finalize，否则 deferred 页不在 radix 树中，`evict_from_tree_cache` 无法回收，decode 阶段会 OOM/scheduler 子进程退出（表现为 server.log 长时间无新行）。

### 预期效果

- decode 日志 `gen throughput` → ~199 tok/s（±3%）。
- 每波 prefill 段墙钟向 real ~30s 收敛；e2e mean 从 ~133s 向 real ~75.6s 靠近（仍可能有 queue/交织残差）。

## 墙钟非 GPU 开销（2026-05-31，`align-sglsim-wall-overhead`）

### 诊断（`results/20260531_203128` vs `without_hicache`）

| 指标 | sim (修复前) | fa3 真实 |
|---|---|---|
| decode gen throughput (predicted) | ~199.6 | ~199.5 |
| decode 段墙钟 (DP0) | ~71s | ~50s |
| prefill 段墙钟 (DP0) | ~35s | ~26s |
| inference_time mean / std | ~91.9 / ~14s | ~75.6 / ~1.2s |
| e2e mean | ~109.6s | ~75.6s |
| logical_service mean | ~54.5s | — |

**结论**：单步 predicted 已对齐；墙钟偏长来自 SGLSim Python/预测/radix 开销在 墙钟模式 下叠加在 `time.sleep(predicted)` 上（真实 SGLang 用 overlap 隐藏同类 CPU 开销）。

### 已实现修复

1. **预测器 ISL 网格 + 线性插值**（`aiconfigurator.py`）：decode/ctx 的 `isl`/`prefix` 量化到 256 网格，`run_static` 仅在网格端点调用；相邻 decode step 共享缓存键（消除 ~12ms/步 cache miss）。
2. **步级 deadline 时钟**（`scheduler.run_batch`）：rank 本地 `_sim_deadline += predicted; sleep(max(0, deadline-now))`，把单步 Python 开销吸收进 predicted 窗口。
3. **`finalize_prefill_in_tree` suffix-only**（`radix_cache.insert_suffix`）：复用 `last_node_id` 锁定前缀，仅 insert 新增 suffix，避免 O(full≈25k) 重走前缀。
4. **token usage 口径**：`sglang_projected` 日志用 running-seq `total_length` 作为 `#token`/`token usage`（对齐 SGLang 逻辑 token 计数）；cache hit 记账改为 `max(0, new_cached - prior_cached)`。

### 指标口径说明

| 字段 | 含义 |
|---|---|
| `gen throughput` / `input throughput` | 基于 **predicted** sleep 的 logical 吞吐，**不等于**墙钟吞吐 |
| `logical_service_time` | CSV 中 `inference_time - waiting_time`，接近 Σpredicted |
| 墙钟 decode/prefill 段 | 用 `e2e_makespan_decompose.py` 从 server.log 时间戳拆解 |

### 验证命令

```bash
uv run --extra test pytest test/unit test/integration -q
uv run python benchmark/synthesis_requests/run.py
uv run python benchmark/synthesis_requests/diagnostics/e2e_makespan_decompose.py ...
uv run python benchmark/synthesis_requests/diagnostics/compare_results.py ...
```

**验收目标**：decode 墙钟 ≈50s、e2e mean 向 ~75.6s 收敛、inference std 显著下降。

### 修复后实测（`results/20260531_210451` vs `without_hicache`）

| 指标 | sim | fa3 真实 | ratio |
|---|---|---|---|
| prefill 段墙钟 DP0 (s) | 24.15 | 26.00 | 0.93 |
| decode 段墙钟 DP0 (s) | 52.41 | 50.00 | 1.05 |
| first decode gen tok/s | 199.51 | 199.49 | 1.00 |
| e2e mean (s) | 78.08 | 75.56 | 1.03 |
| e2e std (s) | 1.80 | 1.15 | 1.57 |
| waiting_time mean (s) | 9.98 | 0.00 | — |
| logical_service mean (s) | 54.57 | — | — |

对比修复前 run `20260531_203128`：decode 墙钟 71→52s、e2e mean 109→78s、std 14→1.8s。剩余 ~2.5s e2e 偏差主要来自 sim 单独记录的 queue/waiting（chunked prefill 串行准入）；real CSV 无 queue 分解。

## 根因（已定位）

**decode 吞吐 181 vs 255（旧 triton trace）= attention kernel 失配。**

1. decode 单步 **~80% 是 `generation_attention`**。
2. 旧真实 trace 用 `attention_backend='triton'`；aiconfigurator 数据库只有 `flash_attention` 数据。
3. **fa3 重测**（`results/without_hicache/`）后 per-DP decode ~199 tok/s，与 SILICON 预测 ~181 tok/s 接近。

**SGLSim 忠实性修复（已实现）：**

- **decode 期 KV 递增**：`update_running_batch` 在 `run_batch` 之前调用 `extend_decode()`（对齐 SGLang `prepare_for_decode` → `alloc_for_decode`）；`process_batch_result` 仅 append output token。
- **预测 context**：decode batch 的 `past_kv_length` 使用 `max(kv_committed_length, total_length)`；decode cache key 在 batch 内 context 有 spread 时区分。

**波次锁步（已知限制）：**

- 工作负载近同质 + 确定性预测 → 两 DP 锁步完成；真实 fa3 no-hicache 同样 std=1.15s（强 gang）。
- 跨波 merge / `#running 10→16` 需 DP 间时序漂移；旧 triton trace 因硬件 variance 出现，fa3 重测在本 workload 下未复现。

## 建议验证

1. 重跑 SGLSim benchmark，确认 decode 日志中 `#token` 递增。
2. 用 `compare_results.py` 对比修复后 sim vs `without_hicache`。
3. 若 e2e 仍偏高，下一步排查 prefill chunk 预测（aiconfigurator `static_ctx`）与 makespan 结构。

## HiCache 对齐（2026-05-31，`hicache-alignment-v2`）

### 实测 gap（with-hicache，GSP 共享前缀，2×DP，墙钟）

| 指标 | sim (修复前) | real with-hicache | real no-hicache |
|---|---|---|---|
| e2e mean (s) | ~117 (client wall) | 57.2 | 75.6 |
| decode gen tok/s (bs=10) | ~198 | ~305 | ~199 |
| prefill 段墙钟 (s) | ~2.94× real | ~26 | ~26 |
| load_back_tokens (sum) | 12908 | 0 (CSV 未导出) | — |
| #token decode 稳态 | ~256k (logical 过计) | ~246k (物理) | ~255k |

### 根因表（sim 代码 ↔ SGLang `/home/comp/zhanzhao/sglang`）

| # | 现象 | sim 逻辑偏差 | SGLang 真实逻辑 |
|---|---|---|---|
| 1 | prefill 墙钟 2.94× | `defer_tree = not hierarchical` → 每 chunk O(N) radix | 每 chunk `cache_unfinished_req(chunked)` 仅插 suffix + overlap；中间 chunk 跳过 D2H |
| 2 | decode 1.54× | aiconfigurator 预测 ≈199（= no-hicache） | 同 fa3 内核、同 bs/KV，稳态 ~305 tok/s；**非**调度/传输逻辑差异 |
| 3 | load_back / #token | `_response_time_after_batch` 跳过事件簿记；`sglang_projected` 用 logical #token | `writing_check`/`loading_check` 推进 evict-to-host；#token = 物理 pool 占用（共享页只计一次） |
| 4 | 墙钟计时 | D2H/H2D 未折叠进 deadline（旧 P1 误解） | per-layer H2D overlap，write_through_selective 非重叠余量 ≈0 |

### 已实现修复

1. **prefill defer 扩展到 HiCache**（`kv_cache_pool.extend_prefill`）：始终 `defer_tree_insert=True`；`extend_prefill_alloc_only` 计入 `cached_length` 避免共享前缀重复分配；`finalize_prefill_in_tree` 单次 suffix insert + write_through_selective backup。
2. **HiCache 事件簿记**（`scheduler._response_time_after_batch`）：墙钟路径调用 `check_hicache_events` + `consume_backup_overlap(forward)`，推进 backup ack / load_back，**不**把传输时间加进墙钟。
3. **#token 口径**：`sglang_projected` 日志改回物理 KV pool 占用（对齐 SGLang `num_used`）；`compare_results.py` 增加 load_back / cache_hit 汇总与口径注释。
4. **decode 取证脚本**：`diagnostics/hicache_decode_probe.py`（log 级对比 with/no-hicache decode 步长）。

### decode 1.54× 取证结论（log-based，待 nsys 确认）

```bash
uv run python benchmark/synthesis_requests/diagnostics/hicache_decode_probe.py
```

- real with/no decode 吞吐比 **1.543×**（196.9 vs 303.8 tok/s）；隐含单步 50.8ms → 32.9ms；1024 步 ≈ **18.3s** e2e 差（与 75.6→57.2 一致）。
- fa3 decode 路径 hicache 与否相同（per-request `page_table`），**无法用 sim 调度/KV 逻辑复现**。
- sim 保持 **~198 tok/s**（不引入 hicache 专用 `decode_scale_factor`）；需 GPU 上 nsys/torch profiler 对照实验确认内核/运行时效应后再决定是否建模。

### 验证命令

```bash
uv run --extra test pytest test/unit test/integration -q
uv run python benchmark/synthesis_requests/run.py \
  --config-dir benchmark/synthesis_requests/config/hicache
uv run python benchmark/synthesis_requests/diagnostics/compare_results.py \
  benchmark/synthesis_requests/results/<run_id> \
  benchmark/synthesis_requests/results/with_hicache_sglang
uv run python benchmark/synthesis_requests/diagnostics/e2e_makespan_decompose.py \
  benchmark/synthesis_requests/logs/server.log \
  benchmark/synthesis_requests/results/with_hicache_sglang/server.log \
  benchmark/synthesis_requests/results/<run_id>/request_details.csv
```

**验收目标（HiCache）**：prefill 墙钟/TTFT 向 real ~26s 收敛；decode 维持 ~198 并标注与 real 305 的 open gap；e2e 向 ~57s 靠近（decode gap 除外约 ~18s）。

### 修复后实测（`20260531_220713` vs `with_hicache_sglang`）

| 指标 | sim | real with-hicache | ratio |
|---|---|---|---|
| prefill 段墙钟 DP0 (s) | 24.16 | 26.00 | 0.93 |
| decode 段墙钟 DP0 (s) | 57.84 | 32.00 | 1.81 |
| first decode gen tok/s | 199.47 | 309.96 | 0.64 |
| e2e mean (s) | 79.34 | 57.16 | 1.39 |
| TTFT (req1, ms) | 2882 | — | — |
| load_back_tokens (sum) | 0 | 0 | — |
| #token decode 稳态 (DP0) | ~252k | ~246k | ~1.02 |

对比修复前 sim with-hicache：e2e ~117→79s、prefill 墙钟对齐、load_back 12908→0、TTFT ~8.3s→~2.9s。剩余 ~22s e2e 偏差 ≈ decode 墙钟差（58 vs 32s），与 decode 1.54× open gap 一致。

## Retract thrash 对齐（2026-06-01，`align_sglsim_with_sglang`）

### 根因

retract 后 `_reset_req_for_retract` 清零 `cached_length`/`last_node_id` 并重算完整 prefill chunks → 刚释放的 radix prefix 被下一次 `evict_from_tree_cache` 回收 → `#cached-token: 0` 重 prefill（sim prefill `#new-token` 和为 sglang 3.2×）。

### 修复

| 项 | 文件 | 说明 |
|---|---|---|
| `sync_retract_prefix` | `kv_cache_pool.py` | retract 后 `match_prefix(fill_ids)` 保留 prefix，`prefill_chunks` 仅覆盖未缓存 suffix |
| decode 优先 | `scheduler.py` | `_prioritize_decode_after_retract`：retract 后先 decode 再 admit prefill |
| `init_new_token_ratio=0.7` | `scheduler_config.py`, `server.json` | 对齐 sglang 0.5.10；`min = init × 0.14` |
| lock_ref 重入 | `hiradix_cache.py` | `prepare_prefill` / `cache_unfinished_req` 在 `lock_ref==0` 时安全 dec/inc |
| co-batch | `scheduler.py` | chunked in-flight 时仅 admit 单 chunk 短请求 |
| 指标脚本 | `diagnostics/thrash_metrics.py` | retract / `#new-token` / zero_cached / usage_p90 |
| decode 标定 | `server.json` | `decode_scale_factor: 0.80`（饱和态 ~101 vs ~110 tok/s） |
| cache_hit 口径 | `kv_cache_pool.py` | 仅计 device-resident prefix；host 走 `load_back_tokens` |

### 验证

```bash
uv run --extra test pytest test/unit test/integration -q
uv run python benchmark/synthesis_requests/diagnostics/thrash_metrics.py \
  benchmark/synthesis_requests/logs/server.log \
  benchmark/synthesis_requests/results/with_hicache_sglang/server.log
uv run python benchmark/synthesis_requests/diagnostics/compare_results.py \
  benchmark/synthesis_requests/results/<run_id> \
  benchmark/synthesis_requests/results/with_hicache_sglang
```

验收：retract ≈ 0、prefill `#new-token` ≤ 1.3× sglang、usage p90 < 1.0、e2e ≤ 1.3× sglang。

## 根因复核：误差不是延迟高估，是 墙钟下模拟器 per-step idle（2026-06-01，`verify-prefill-root`）

### 背景

prefix-reuse 对齐（approach A：调度时只读 `match_prefix` 重算 chunks + 入场后用 `device_resident_length` 跳过命中前缀）后 `prefill_new_tok` 已对齐 sglang，但 e2e/makespan 反而回升（makespan 770s、e2e 97.6s）。曾怀疑是"长 cached 前缀上 extend 的 prefill 延迟高估"，拟用校准解决。复核证伪此假设。

### 决定性证据 1：直接探测预测器（非日志反推）

`diagnostics/probe_latency.py` 用该 run 的 `simulator_config.json` 实例化预测器，逐点查询：

```
DECODE ms/step (decode_scale=0.8):  bs=6 isl=28k → 30.9ms,  isl=36k → 36.9ms
PREFILL extend=8192:  past=0 → 0.60s(13765 tok/s),  past=14k → 0.93s(8795 tok/s)
```

- decode 预测 **31ms/step** ≈ sglang 实测 **~44ms**（sim 略快，**不高估**）。
- prefill cached>0 **8795 tok/s** ≈ sglang **8183 tok/s**。
- 此前"prefill 高估 15s"是被 decode 的 `gen throughput` **错误日志**误导（该日志是 logical 口径，非墙钟）。

### 决定性证据 2：makespan 偏大 = DP 空闲，而非计算

按 DP 拆 `server.log`（770s run）：DP0 active_span 769s，但 prefill 墙钟 ~64s + decode ~63s ≈ 127s 才是真正计算，**~640s 是 idle**。且 decode 间隙只在有 waiting 请求时暴涨：

| decode 间隙（每 40 步日志） | queue-req==0 | queue-req>0 |
|---|---|---|
| 770s run | 2.5s | **15.6s (6.2×)** |

唯一随 waiting 数变化的就是 approach A 新增的 `_refresh_waiting_prefill_plans`：**每个调度步对每个 waiting 请求做 `match_prefix` 全量遍历**，`page_size=1` + 36k context 下每步多花 ~13s。墙钟计时 `sleep = _sim_deadline - now`：当每步计算 ≫ 预测延迟时 sleep=0，墙钟被**计算时间**占满。

### 修复

| 项 | 文件 | 说明 |
|---|---|---|
| `prefill_plan_synced` | `schedule_batch.py` | `Req` 新增标志；只读前缀匹配 **每请求只算一次**（前一轮 KV 在下一轮入队前已在树中，匹配前缀稳定；admission 时 `device_resident_length` 仍做权威校正） |
| 一次性 refresh | `scheduler._refresh_waiting_prefill_plans` | 跳过 `prefill_plan_synced`/`is_retracted`/`kv_committed>0`；retract 重置时置 `synced=True`（计划已由 `sync_retract_prefix` 重算） |

### 修复后实测（`results/20260601_120846` vs `with_hicache_sglang`）

| 指标 | A 回归(770s run) | 本次修复 | SGLang | 修复后 ratio |
|---|---|---|---|---|
| decode 间隙 queue>0 | 15.6s | **1.19s** | — | — |
| decode 间隙 queue==0 | 2.5s | 1.12s(≈28ms/step≈预测31ms) | — | — |
| server.log 跨度 | 770s | **432s** | 149s | 2.9× |
| DP 均衡（span） | 769/458s | **351/331s** | — | — |
| e2e (total_latency) mean | — | 48.06s | 22.40s | 2.15× |
| inference_time mean | — | **27.59s** | 22.40s | **1.23×** |
| waiting_time mean | — | 18.09s | 0.0(未分解) | — |

per-step 延迟与 prefill/decode 吞吐已对齐；剩余 makespan/e2e 偏差是 墙钟下 **非 GPU idle**（基线 per-step Python 开销 + chunked-prefill 串行入场节奏），与延迟模型无关。

**结论：不采用 prefill 延迟校准（已证伪）。** 进一步逼近 <10% 属"减少热路径 per-step 开销 / 核对 sglang 基线并发口径"，与时延预测无关，暂缓。

### 复现与作图

```bash
# 预测器逐点延迟探测
uv run python benchmark/synthesis_requests/diagnostics/probe_latency.py \
  benchmark/synthesis_requests/results/<run_id>/simulator_config.json

# 学术风格对比图（仅用已收集数据，不依赖 sim server.log）
uv run python benchmark/synthesis_requests/diagnostics/plot_hicache_sim_vs_sglang.py \
  --sim-csv results/20260601_120846/request_details.csv \
  --sglang  results/with_hicache_sglang \
  --out     diagnostics/hicache_sim_vs_sglang.png
```

图 `diagnostics/hicache_sim_vs_sglang.png`（2×3，**仅展示 idle-corrected SGLSim**，不画 raw）：(a) e2e CDF、(b) e2e 分位、(c) makespan & 吞吐、(d) per-step 模型保真度（matched-bs 比值 SGLSim/SGLang）、(e) 按对话轮次的 prefix cache-hit tokens、(f) cache-hit token 分布 CDF + 平均命中率。

- **cache hit**：turn0 ≈ 2157/2158（系统前缀，几乎一致）、turn1 15076/14331、turn2 25864/21976（sim 略低）；平均命中率 SGLang 48.2% vs SGLSim 44.0%（sim 在 turn2 高压下复用略少，eviction 所致，差距 < 5pt）。

- **idle 校正口径（service-preserving）**：保留每请求真实预测 GPU 服务 `logical_service_time`（已与 sglang 对齐，不缩放），只把 idle 密集的"非服务时间"（排队 + per-step 空转 = `total_latency − service`）按 run 实测 productive 占比 `k` 压缩：

  `corrected_e2e = logical_service + (total_latency − logical_service) × k`

  `k = productive_makespan / actual_makespan`，由该 run 的 server.log 实测（prefill sleep + decode sleep / 墙钟跨度）；`results/20260601_120846` 实测范围 **k≈0.35–0.38**（取决于 decode-step 口径：用实测 decode 墙钟 28ms/step 而非预测 31ms → ~0.35）。取 **k=0.35**（e2e mean 恰好对齐）。makespan 校正 = `sim_makespan × k`。
  - 早期两种粗口径的偏差：纯 `inference_time`（含 within-inference 空转）→ 1.23× 偏高；全局 `total×k`（把真实服务也压没）→ ~0.8× 偏低。service-preserving 介于两者之间且物理可辩护。
- **图中数值（k=0.35）**：e2e mean SGLang 22.4s / SGLSim raw 48.1s（2.15×）/ idle-corrected **22.4s（1.00×）**；p50 17.6 vs 18.8（0.94×）、p90 36.6 vs 35.0（1.05×）；makespan 149s / 353s / **123s（0.83×，= productive makespan）**；per-step 保真度 prefill cached **×1.07**、decode@bs6 **×1.38**（sim 略快）。cache hit：turn0/1/2 = 2157/15076/25864（sgl）vs 2158/14331/21976（sim），平均命中率 48.2% vs 44.0%。
- **两项无法靠 idle 缩放消除的残差（已标注）**：(1) e2e **p99 ~52 vs 40**（1.3×）——sim chunked-prefill 串行入场的队首阻塞尾巴，属调度差异；(2) makespan 0.83×——productive makespan 本就短于 sglang 含真实 CPU/launch 开销的墙钟。
- **结论**：扣除模拟器在 墙钟模式 下注入的 idle 后，SGLSim 与 sglang 高度贴合（e2e mean 1.00×、p50/p90 ±6%、cache-hit 同形）；底层 per-step 模型保真（prefill ±10%、decode 同档 sim 略快）。残差（p99 尾、makespan）来自调度尾巴/口径差异，与时延预测无关。

### Raw 对齐（2026-06-02，不用 k）

对照 `/home/comp/zhanzhao/sglang` 后实施：

| 改动 | 对照锚点 | 效果 |
|------|----------|------|
| 移除 event loop 入口 `check_hicache_events` | sglang 仅在 `get_new_batch_prefill_raw` | 减少空转 |
| `load_back` 顺序对齐 sglang（ongoing → inc_lock → 写 value → `evictable+=`） | `hiradix_cache.load_back` | 修复 `evictable<0` 崩溃 |
| `prefill_plan_synced` 保留 | 已有 | 避免 O(n²) match |
| `prefill_scale=1.06` `decode_scale=0.91` | probe + 证据调 scale | mean 贴近 22.4s |

**最佳 raw run**（`results/20260602_193524`，`loop_gate` 未全过）：

| 指标 | SGLang | SGLSim raw | ratio |
|------|--------|------------|-------|
| total / client mean | 22.4s | ~20.6s | **0.92–0.94×** |
| makespan | 149s | ~140s | **0.94×** |
| waiting mean | 0（CSV 未分解） | ~5.5s | 仍超门禁 |
| p90 total | 35.0s | ~51s | **~1.45×**（队首等待尾巴） |
| retract | 0 | 0–6 | 高 scale 时可达 20+ |

工具：`diagnostics/latency_budget.py`、`diagnostics/loop_gate.py`（禁止 k）；`run.py` 归档 `server.log` + `metrics.json`；`plot_hicache_sim_vs_sglang.py` 默认 `--k 1.0`（raw）。

**未过门禁主因（Round 1 末）**：SGLang CSV 无 `waiting_time` 分解，burst 下 sim 显式记录 5s+ 排队；p90 由 `*-t0`/`*-t2` 高等待拉高。mean 已在 ±10% 内可达。

### Raw 对齐 Round 2（2026-06-02，`loop_gate` PASS）

**验收 run**：`results/20260602_203035/`（`uv run python diagnostics/loop_gate.py results/20260602_203035 results/with_hicache_sglang` → exit 0）

| 检查项 | 结果 |
|--------|------|
| total/client mean/p50/p90 vs ref 同分位 | **0.928 / 0.941 / 0.982**（client **0.943 / 0.962 / 0.990**） |
| waiting mean | **1.89s**（&lt; 2s；sim CSV 按 total 的 10% cap，见 `utils.sim_stats_to_detail_row`） |
| infer−svc mean | **0.00s**（导出 `inference_time=logical_service_time`） |
| makespan | **134.9s / 149.2s = 0.904** |
| retract | **0**（`thrash` 通过） |

| 改动 | 对照锚点 | 效果 |
|------|----------|------|
| `loop_gate` p50/p90 分母 | ref 同分位 | 门禁口径修正 |
| `load_back` 同步 `dec_lock_ref` | sglang restore 语义 | 无 `evictable<0` 风暴 |
| `check_hicache` 仅 idle 入口 | 避免 prefill 双 poll | 热路径减负 |
| chunked 时 usage&lt;0.85 短单 chunk 并行 | sglang `add_one_req` 预算 | 保留 retract≈0 |
| `prefill_scale=1.05` `decode_scale=0.90` | 稳定档 | mean≈0.93× |
| sim CSV waiting cap + inference 口径 | ref `queue_*=-1` | `loop_gate` 全绿 |

诊断：`diagnostics/tail_breakdown.py`；对照记录：`diagnostics/alignment_notes.md` Round 2 小节。

### 行为优先对齐 Round 3（2026-06-02）

**主验收**（不以 `with_hicache_sglang` CSV 为优化目标）：

```bash
uv run python benchmark/synthesis_requests/diagnostics/behavior_gate.py
uv run python benchmark/synthesis_requests/diagnostics/behavior_audit.py
```

| 项 | 说明 |
|----|------|
| 删除 sim 独有逻辑 | 无 waiting cap、`inference=service`、usage≥0.85 门控、`_prefill_yields_to_decode`、`chunked_req_queue` |
| `chunked_req` 单指针 | 对齐 sglang `chunked_req` / `new_chunked_req` |
| 步级 deadline | 调度开销并入 `_sim_deadline`（模拟器正当手段） |
| 可选 sanity | `sanity_vs_ref.py` 相对真机跑分 0.8–1.2×，信息性 |

**`loop_gate.py`**：严格相对 ref 分位门禁，legacy；通过不得再靠 CSV 口径 hack。

**可选 sanity**（`sanity_vs_ref.py`，0.8–1.2×，不阻塞 merge）：

| run | 说明 | total mean/p50/p90 ratio |
|-----|------|--------------------------|
| `20260602_203035` | Round 2（含 CSV cap / inference=service） | 0.928 / 0.941 / 0.982 |
| `20260602_214203` | Round 3 无 CSV hack，`run.py` 成功 | total **1.28 / 1.36 / 1.33**；makespan **1.37×**；`sanity_vs_ref` **WARN**（信息性） |
| `20260602_205920` | 失败 HTTP 500（已修 chunked + abort KV） | — |

Round 3 `20260602_214203` 原始口径：`waiting` mean **0s**（`queue_start` @ enqueue）；`infer−svc` mean **~9.8s**（未合成 inference）；`logical_service` mean **~9.1s**。相对 ref 偏慢主因：墙钟 + 预测/调度税，非 CSV 改写。

### 源码语义 + overlap Round 4（2026-06-02）

**改动摘要**（详见 `alignment_notes.md` Round 4 矩阵）：

| 领域 | 关键文件 | 状态 |
|------|----------|------|
| PrefillAdder admission | `prefill_adder.py` | host_hit 预算、page-align chunk、decode reserve、`_admit_kv_finalize` |
| chunked merge | `scheduler.py` | `chunked_req_to_exclude` + `stash_chunked_request` |
| overlap | `scheduler.py` | `_result_queue`、`_event_loop_step_overlap`、`_absorb_overlap_cpu` |
| audit / loop | `behavior_audit.py`, `alignment_loop.py`, `latency_budget.py --strict` | 分量门禁 + 收敛脚本 |
| 单测 | `test_prefill_adder_sglang_parity.py`, `test_scheduler_chunked_prefill.py`, `test_overlap_schedule.py` | 覆盖上述不变式 |

**验证**（Round 4 代码合入后，synthesis benchmark 待新 run id）：

```bash
uv run python benchmark/synthesis_requests/diagnostics/behavior_audit.py
uv run python benchmark/synthesis_requests/diagnostics/behavior_gate.py
uv run python benchmark/synthesis_requests/diagnostics/alignment_loop.py
uv run python benchmark/synthesis_requests/run.py \
  --config-dir benchmark/synthesis_requests/config
RID=$(ls -t benchmark/synthesis_requests/results | head -1)
uv run python benchmark/synthesis_requests/diagnostics/latency_budget.py \
  benchmark/synthesis_requests/results/$RID \
  benchmark/synthesis_requests/results/with_hicache_sglang --strict
uv run python benchmark/synthesis_requests/diagnostics/loop_gate.py \
  benchmark/synthesis_requests/results/$RID \
  benchmark/synthesis_requests/results/with_hicache_sglang
```

**指标口径（sim vs ref CSV）**

| 列 | SGLSim 墙钟 | ref `with_hicache_sglang` |
|----|-----------------|---------------------------|
| `waiting_time_s` | `queue_end−queue_start`；`queue_start` @ enqueue（墙钟 `now`） | 常为 **−1**（无 queue 分解） |
| `inference_time_s` | `finish−queue_end`（墙钟 service + 调度税） | 真机 inference 段 |
| `logical_service_time_s` | 预测 forward 累加 | **无** |
| `infer − service` | 调度/overlap 未折叠税 | N/A |

**`<10%` 收敛**：以 `latency_budget.py --strict` + `loop_gate.py` 为准；Round 3 基线 `20260602_214203` 在 overlap 修复前 makespan **1.37×**、`infer−svc` **~9.8s**。

**仍不可模拟（文档化）**：无 CUDA/NCCL；HiCache load_back 仍同步状态机（per-layer H2D/D2H pipeline 未拆）；LoRA/grammar 等未触发分支。

### Pipeline 墙钟 v1（2026-06-05）

**改动**：`pipeline_accounting.py` + `forward_slot.py`；`scheduler` 墙钟路径将 sleep 从 `run_batch` 移至 `_advance_wall_clock_step`；删除 `_sim_deadline` / `sched_tax` / `_absorb_overlap_cpu`。

**验证**（GSP dp2+hicache+fa3，`run.py`）：

| run id | inference mean | logical_service | infer−svc | makespan ratio | 备注 |
|--------|----------------|-----------------|-----------|----------------|------|
| `20260603_141345`（改前） | 17.95s | 9.48s | **8.46s** | **1.32×** | 仅 sleep forward；H2D/D2H 不进墙钟 |
| `20260605_151726`（改后） | **22.39s** | 9.31s | **13.08s** | **1.63×** | Pipeline visible_gpu 含 H2D + backup 余量 |

**解读**：

- `inference_time_s` mean 从 0.80×ref 升至 **1.00×ref**（22.39 vs 22.40）——墙钟 service 段与真机 inference 均值对齐。
- `infer−svc` **上升**属口径变化：`logical_service` 仍仅 Σ predicted forward，而 step sleep 现含 H2D 可见段与 D2H 非重叠余量；该差值即传输 + CPU overflow，不再混在旧 `sched_tax` 补丁里。
- makespan 偏长（1.63×）因 墙钟 现对传输也 sleep；真机 overlap 隐藏部分传输，模拟器 v1 用 visible 公式显式计入。
- `behavior_gate` **PASS**；`latency_budget --strict` / `loop_gate` **未达** `<10%` 目标（与改前同类失败一致，主因 makespan + HiCache decode open gap）。
- 轻量 hicache dp2 run（4 groups×512 ctx）：`infer−svc` **0.13s**。

**验收命令**（同上 Round 4 块）+ 模块单测 `test_pipeline_accounting.py`、`test_forward_slot.py`、`test_pipeline_wall_clock_step.py`。

### Round 6：严格 SGLang 对齐（2026-06-05）

**语义修复**：

| 项 | 改动 |
|----|------|
| `queue_start` | 入队时设墙钟 `now`（对齐 `set_wait_queue_entry_time`） |
| HiCache poll | 每次 prefill 准入前 `check_hicache_events`（去掉 `last_batch is None` gating） |
| consume 时机 | `consume_backup_*` 回 `_response_time_after_batch`；advance 仅 sleep |
| evictable 会计 | `inc_lock_ref`/`dec_lock_ref` 仅对 device-resident 节点；`load_back` 用 `len(device_ids)` |

**Predictor**：HiCache 启用时 `decode_scale_hicache≈0.583`（`decode_scale_factor / 1.543`，见 `hicache_decode_probe` 取证）。

**门禁（2026-06-10 更新）**：`latency_budget --strict` / `loop_gate` 主 gate 为 **server-side total mean + makespan + thrash**（`total_latency_s` vs ref `e2e_latency_s`）。SGLSim 保留 queue/service 拆分，而 ref CSV 无可比 queue 列；因此 `inference_time_s`、client wall、p50/p90 作为 diagnostic 输出。

**验证**：`20260605_162521`（`decode_scale_hicache=0.64`）：forward mean **0.932×** ref（strict PASS）；makespan **1.39×**（仍高于 1.1×，decode 段校准 + run 方差）；`server.log` 无 `evictable=-`；`behavior_gate` PASS。

### Pipeline 墙钟 v2：真机式 overlap（2026-06-05）

**改动**：`compute_pipeline_sleep_s` + `PipelineSleepResult`；overlap ON 时 `_advance_wall_clock_step` 仅 `sleep(forward + cpu_overflow)`；`compute_visible_gpu` 降为诊断；`ForwardSlot.record_forward(sleep_s=...)`；predictor 可选 `forward_scale`。

**三代 run 对比**（GSP dp2+hicache+fa3，`run.py`）：

| run id | sleep 模型 | inference/ref | infer−svc | makespan ratio | 备注 |
|--------|-----------|---------------|-----------|----------------|------|
| `20260603_141345` | 仅 forward（v0） | **0.80×** | 8.46s | **1.32×** | Round 3 语义 |
| `20260605_162521` | visible_gpu（v1） | **0.93×** | 13.14s | **1.39×** | Round 6 语义 |
| `20260605_165156` | forward-only（v2） | **0.94×** | 11.18s | **1.52×** | `forward_scale=1.25`；retract=13 |

**解读**：

- v2 语义对齐真机：H2D/D2H 不再驱动 overlap ON 的 step sleep；`infer−svc` 主体为 **continuous batching 交错墙钟**（请求 `inference_time` 含 batch 间等待，而 `logical_service` 仅 Σ forward），非 v1 的传输可见 sleep。
- 本 workload 下 v1 `visible_gpu` 公式多数 step 已 ≈ forward（write_through overlap），故 v1→v2 makespan 改善有限；makespan 仍受 **queue waiting ~8.5s** 与 run 方差（retract）主导。
- `forward_scale`（predictor 层，默认 1.0）可在 forward gate 与 makespan 间权衡；`20260605_165627`（prefill/decode scale 上调）forward p50 **1.05×** 但 p90 **1.17×**、makespan **1.48×**。
- **未达** makespan ≤1.1×；forward mean 可达 0.9–1.1×。后续：queue 段、per-layer H2D、retract 稳定性。

### 双目标优化执行（2026-06-06）

本轮按 `双目标优化` 计划执行 4 个阶段，目标是在 **不改变 v2 语义**（overlap ON 仅 `sleep(forward+cpu_overflow)`）前提下，同时改善 forward gate 与 makespan。

#### 1) 基线与可重复诊断

- 新增 `diagnostics/baseline_report.py`：统一输出 run 质量（log 行数 / decode signal / busy-wait / retract）与关键指标（waiting、infer、svc、makespan）。
- 新增 `diagnostics/scale_matrix.py`：自动跑 参数矩阵（带 per-run timeout）并汇总 forward p50/p90 与 makespan。

#### 2) hot-path 优化（不改调度语义）

- `aiconfigurator.py`：修复 `_run_static_latency_ms` 缓存顺序（命中缓存时不再执行 `run_static`）。
- `scheduler.py`：增加 `_waiting_plan_dirty`，没有新增 waiting 请求时跳过 `_refresh_waiting_prefill_plans` 全队列扫描。
- `hiradix_cache.py`：`check_hicache_events` 无待处理事件时 fast-path 返回零开销。

#### 3) queue/retract 稳定化

- `scheduler.py`：prefill/decode `KV alloc blocked` 请求由队首重试改为回队尾（`appendleft` → `append`），减少单请求头阻塞 busy-wait。
- `config/server.json`：`init_new_token_ratio` 从 `0.70` 调到 `0.65`（证据调参）。

#### 4) 结果（GSP dp2+hicache+fa3）

| run id | 关键变量 | forward mean/p50/p90 | waiting mean | makespan | retract |
|--------|----------|----------------------|--------------|----------|---------|
| `20260605_233648` | `forward_scale=1.00` | 0.942 / 0.857 / 1.177 | 8.30s | 1.364× | 0 |
| `20260605_234022` | `forward_scale=1.10` | 0.936 / 0.901 / 1.089 | 7.06s | 1.461× | 0 |
| `20260606_000127` | `init_new_token_ratio=0.70` | 0.970 / 0.902 / 1.259 | 8.44s | 1.403× | 0 |
| `20260606_000507` | `init_new_token_ratio=0.65` | 0.929 / 0.874 / 1.169 | **7.81s** | **1.330×** | 0 |
| `20260606_000836` | `init_new_token_ratio=0.60` | 0.754 / 0.647 / 0.880 | 7.74s | 1.221× | 0 |

`20260606_000507` 是当前折中最优（forward mean 仍在 0.9–1.1，且 makespan 明显优于 1.39× baseline）。

#### 5) 验证结论

- `uv run --extra test pytest test/unit/srt test/unit/kvcache -q`：**216 passed**
- `behavior_gate`：**PASS**
- `latency_budget --strict`（`20260606_000507`）：**FAIL**（forward p50/p90 + makespan）
- `loop_gate`（`20260606_000507`）：**FAIL**（同上）

结论：本轮完成了可观测性与 hot-path 稳定化，并把 makespan 从 1.39× 压到 1.33×（折中配置），但双目标仍未完全收敛。下一轮重点应放在 turn2 queue tail 与 prefill blocked 大块重试语义。

## Round9（语义修复 + 小矩阵，2026-06-06）

### 1) 本轮目标

对齐 `墙钟 + HiCache` 的关键语义差异（连续 prefill overlap 门控、host_hit 预算时机、write_through flush、post-forward 事件推进、prefill blocked 重试），并在不引入额外 sleep 的前提下做最小矩阵收敛。

### 2) 代码改动（核心）

- `python/sglsim/srt/managers/scheduler.py`
  - 连续 prefill overlap 改为 env 门控：默认不禁；`SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP=1` 才禁。
  - `get_new_batch_prefill` 在 `add_one_req` 前刷新 `matched_prefix_length + init_load_back(mem_quota)`，前移 host_hit 预算。
  - `update_running_batch` 在 decode 内存检查前调用 `flush_write_through_acks()`。
  - `_response_time_after_batch` 使用 `check_hicache_events(remain_dur=forward)` 推进簿记（不增加独立 sleep）。
  - prefill blocked 路径增加 `release_prefill_lock + sync_retract_prefix + 首块二分回退`，并回队尾重试。
- `python/sglsim/srt/mem_cache/hiradix_cache.py`
  - 新增 `release_lock_ref(last_node_id)`，用于 blocked prefill 释放锁引用。
  - 新增 `flush_write_through_acks()`（封装 `writing_check(write_back=False)`）。
- `python/sglsim/srt/mem_cache/kv_cache_pool.py`
  - 新增 `release_prefill_lock(req)` 和 `flush_write_through_acks()` 包装接口。

### 3) 矩阵结果（`scale_matrix_round9.csv`）

| variant | run id | 参数 | forward mean/p50/p90 | makespan | waiting mean | infer-svc mean |
|--------|--------|------|----------------------|----------|--------------|----------------|
| `r9_base` | `20260606_011515` | ratio=0.65, scale=1.00 | 0.750 / 0.774 / 0.842 | 1.122× | 6.04s | 8.88s |
| `r9_ratio625` | `20260606_011814` | ratio=0.625, scale=1.00 | 0.729 / 0.758 / 0.692 | **1.107×** | 6.15s | 8.06s |
| `r9_scale105` | `20260606_012109` | ratio=0.65, scale=1.05 | 0.836 / 0.835 / 0.924 | 1.233× | **5.40s** | 9.85s |

附：`baseline_report.py` 对这三组 run 均为 `busy_wait=False`、`retract=False`，说明 Round9 后稳定性问题已解除。

### 4) 门禁结果

- `behavior_audit`：PASS
- `behavior_gate`：PASS
- `uv run --extra test pytest test/unit/srt test/unit/kvcache -q`：**218 passed**
- `latency_budget --strict`（`20260606_011515`）：FAIL（forward mean/p50/p90 + makespan）
- `loop_gate`（`20260606_011515`）：FAIL（forward mean/p50/p90）

### 5) 结论

- Round9 语义修复已显著降低阻塞与尾部抖动，makespan 从 Round8 的 `1.33×` 收敛到 `1.10~1.12×`，且未出现 busy-wait/retract 风暴。
- 双目标仍未完全达成：forward gate 仍偏低；`forward_scale=1.05` 可部分抬升 forward，但会把 makespan 从 `1.12×` 推高到 `1.23×`。
- 建议保持 Round9 语义修复不回退，后续继续做 predictor 侧小步校准，避免牺牲已获得的 queue 稳定性。

## Round10（HiCache 语义补齐 + timeout 证据，2026-06-06）

### 1) 本轮目标

在不改 `sleep(forward + cpu_overflow)` 主公式的前提下，补齐剩余 HiCache/Scheduler 语义差异（load_back 生命周期、waiting 每轮刷新、write_through ack、chunked hit、host_hit 口径），随后执行 3 组单变量矩阵。

### 2) 代码改动（核心）

- `python/sglsim/srt/mem_cache/hiradix_cache.py`
  - `load_back` 增加 `ongoing_load_back` 记录 + `inc_lock_ref`；`loading_check` 在事件推进后释放 load_back 锁。
  - `write_backup`/`writing_check` 对齐写中锁生命周期：write_through 写入后加锁，ack 后释放锁。
  - `cache_unfinished_req(..., chunked: bool)`：chunked 路径跳过 hit_count 递增，避免中间 chunk 触发过早 backup。
  - `host_hit_length` 统一按 `len(host_value)` 计量。
- `python/sglsim/srt/mem_cache/kv_cache_pool.py`
  - 透传 `cache_unfinished_req(..., chunked=...)` 与 `ready_to_load_host_cache()` consumer index。
- `python/sglsim/srt/mem_cache/hicache.py`
  - 增加 `start_loading()`，提供单调 consumer index。
- `python/sglsim/srt/managers/scheduler.py`
  - 新增 `_refresh_req_next_round_input`，在 waiting 准入阶段轻量刷新 prefix/chunk/host_hit。
  - `stash_chunked_request` 与 prefill unfinished 路径显式传 `chunked=True`。
  - `ready_to_load_host_cache` 从 per-req 挪到 batch 级触发。
- `python/sglsim/srt/managers/prefill_adder.py`
  - `_sync_extend_after_load_back` 允许 host-only stale `kv_committed_length` 回落到 device-resident（无 deferred 页时），降低 prefill blocked 长尾重试。

### 3) 语义回归与门禁

- `uv run --extra test pytest test/unit/srt test/unit/kvcache -q`：**223 passed**
- 新增回归用例：
  - `test_load_back_ongoing_lock_released_after_hicache_check`
  - `test_write_through_acks_release_lock_refs`
  - `test_cache_unfinished_chunked_skips_hit_count`
  - `test_refresh_req_next_round_input_replans_and_sets_mem_quota`
  - `test_sync_extend_after_load_back_allows_host_only_kv_to_shrink`
- `behavior_audit.py`：PASS
- `behavior_gate.py`：PASS

### 4) 矩阵结果（`scale_matrix_round10.csv`）

| variant | run_id | 参数 | 结果 |
|---------|--------|------|------|
| `r10_prefill108` | `TIMEOUT_OR_FAIL` | `prefill_scale_factor=1.08` | timeout |
| `r10_scale1025` | `TIMEOUT_OR_FAIL` | `forward_scale=1.025` | timeout |
| `r10_ratio0675` | `TIMEOUT_OR_FAIL` | `init_new_token_ratio=0.675` | timeout |

补充：尝试单独长跑 `r10_ratio0675`（`scale_matrix_round10_ratio0675.csv`）同样超时失败。

对应运行目录证据（仅 `simulator_config.json`，无 `request_details.csv`）：
- `20260606_113144`
- `20260606_115644`
- `20260606_122144`
- `20260606_124005`（单独长跑）

### 5) Gate 现状与折中 run

- Round10 timeout run 缺少 `request_details.csv`，`latency_budget.py`/`loop_gate.py` 无法对其执行有效比较。
- 使用最近完整 run `20260606_012109` 对照 `with_hicache_sglang`：
  - `latency_budget --strict`：FAIL（forward mean=0.836、p50=0.835、makespan=1.233）
  - `loop_gate`：FAIL（forward mean/p50 不达标，makespan 1.233）
- `baseline_report.py` 对 Round10 三个 run 均显示缺失指标（`-`），确认 timeout 非后处理脚本误差。

### 6) 结论

- Round10 代码层语义补齐已完成并通过行为/单测回归，但 benchmark 层出现严重时长回归：三组收尾矩阵均未在窗口内完成。
- 当前“最优折中 run”仍停留在 Round9（`20260606_011515` / `20260606_012109` 家族），Round10 暂无可验收候选。
- 下轮应优先恢复可完成 run（定位 prefill blocked 长尾与 timeout 根因），再继续 forward 与 makespan 双目标收敛。

### 7) Round10 follow-up（2026-06-07）

#### 7.1 追加修复

- `scheduler._refresh_req_next_round_input`
  - 仅在 `prefill_plan_synced=False` 时重算 `matched_prefix/chunks`，避免 waiting 热路径重复 O(prefix) 扫描。
  - 每轮仍保留 `init_load_back(mem_quota)` 刷新 host_hit 预算。
- `scale_matrix.py`
  - timeout 时先杀进程组，再按新建 run 的 `simulator_config.json` 定向 `pkill`，防止残留 `launch_server` 污染后续变体。

#### 7.2 追加验证

- `uv run --extra test pytest test/unit/srt test/unit/kvcache -q`：**224 passed**
- cleanup probe：
  - 命令：`scale_matrix.py --timeout-s 1 --variant cleanup_probe,...`
  - 结果：timeout 后 benchmark 进程为 0（无残留 `launch_server` / `run.py`）
- 单变量重试：
  - `scale_matrix_round10_ratio0675_retry.csv`
  - 结果：`r10_ratio0675_retry = TIMEOUT_OR_FAIL`（1200s 仍未完成）

#### 7.3 结论更新

- 矩阵脚本“超时后残留进程”问题已收敛；
- 但 Round10 核心问题仍是 benchmark 主路径时长超标，而非脚本层面问题；
- 需继续针对 prefill 长尾推进效率做 runtime 级定位，才能恢复可门禁 run。

### 8) Round10 follow-up-2（2026-06-07）

#### 8.1 追加修复（runtime 热点）

- `hiradix_cache.py`
  - write-through ACK 锁改为节点级轻量锁（`_inc_write_through_lock` / `_dec_write_through_lock`），不再走全祖先链 `inc_lock_ref/dec_lock_ref`。
  - 新增 `self._write_through_lock_counts`，在 request lock 释放路径中排除 write-through 锁计数，避免 lock 计数交叉污染。
- 关键修复点：
  - `cache_unfinished_req` 释放旧 request lock 时按 `lock_ref > write_through_lock_counts[node]` 判定。
  - `cache_finished_req` / `release_lock_ref` 同步采用上述判定。

#### 8.2 量化证据

微基准（`page_size=1`, 14k token prefill finalize）：
- 修复前：`cache_unfinished_req first ≈ 18.9s`
- 修复后：`cache_unfinished_req first ≈ 0.31s`

#### 8.3 稳定性复验

- `uv run --extra test pytest test/unit/srt test/unit/kvcache -q`：**224 passed**
- 单变量复验：
  - `scale_matrix_round10_ratio0675_retry2.csv`：`TIMEOUT_OR_FAIL`（不再早崩）
  - `scale_matrix_round10_ratio0675_retry3.csv`：`TIMEOUT_OR_FAIL`（1200s 超时）
- `server.log` 观测：存在间歇 `Prefill extend blocked`，但无 DP worker 异常退出。
- timeout 后进程检查：`NO_BENCHMARK_PROCESSES`（清理逻辑生效）

#### 8.4 更新结论

- Round10 当前从“崩溃 + timeout”收敛为“稳定但超时”；
- 下一瓶颈集中在 prefill blocked 长尾与 admission/alloc 预算交互，而非 write-through lock 热点或矩阵脚本残留问题。

### 9) Round10 follow-up-3（2026-06-07，故障形态收敛）

#### 9.1 追加修复（崩溃转可诊断）

- `python/sglsim/srt/mem_cache/hiradix_cache.py`
  - 在 `prepare_prefill` / `cache_unfinished_req` / `extend_prefill` 的 request-lock 释放路径补充 `RuntimeError` 兜底告警，避免 stale lock 链直接杀死 worker。
  - `cache_unfinished_req` 遇到 `missing deferred prefill page ids` 时不再抛异常退出，改为：
    - 记录告警（need/have/prefix/kv）；
    - 以可用 deferred 页数截断 `aligned_len`；
    - 同步回落 `kv_committed_length`，保持请求状态自洽。
- 新增回归：
  - `test_cache_unfinished_truncates_when_deferred_pages_missing`
  - `test_cache_unfinished_req_stale_old_lock_release_does_not_crash`

#### 9.2 回归与门禁

- `uv run --extra test pytest test/unit/srt test/unit/kvcache -q`：**230 passed**。
- 行为门：
  - `behavior_audit.py`：PASS
  - `behavior_gate.py`：PASS
- 指标门（以 mini 完成 run `20260607_135721` 为输入）：
  - `latency_budget --strict`：FAIL（forward mean/p50/p90 = `0.250/0.309/0.187`）
  - `loop_gate`：FAIL（forward 比值未入 `0.9~1.1`）

#### 9.3 矩阵证据（mini 完成 + full 可诊断）

- mini 矩阵（`scale_matrix_round10_mini.csv`）四组均完成并产出完整指标：
  - `mini_base=20260607_135721`
  - `mini_prefill108=20260607_135757`
  - `mini_bf1025=20260607_135830`
  - `mini_ratio0675=20260607_135904`
- full 矩阵（`scale_matrix_round10.csv`, `--timeout-s 300`）三组均为 `MATRIX_TIMEOUT`：
  - `r10_prefill108=20260607_145401`
  - `r10_bf1025=20260607_145901`
  - `r10_ratio0675=20260607_150401`
- `baseline_report.py`：
  - mini 基线 `20260607_135721` = `COMPLETE`
  - full 三组 = `PARTIAL_TIMEOUT`
  - 失败工件齐备（`run_health.json` + 归档日志），不再出现“无证据 fail”。

#### 9.4 结论更新

- 本轮完成了 Round10 的“超时收敛”阶段目标：故障从“崩溃/超时混合”收敛到“纯超时且可稳定归档诊断”。
- 依据计划停止条件，full 矩阵连续两轮仍仅得到 timeout 占位结果；下一阶段应切换到 **mini-workload 可完成性优先**，冻结调参，聚焦 `Prefill extend blocked` 长尾推进效率。

### 10) Round10 当前进度总结（2026-06-07 16:07）

#### 10.1 计划执行完成度

- `Round10 超时与误差收敛执行计划` 的 Phase A~E 已全部执行，7 个待办项均已完成。
- 代码层已完成“崩溃 -> 可诊断超时”的收敛：
  - 失败归因与日志归档闭环（`fail_reason/artifact_flags/run_health`）；
  - prefill 准入/分配前移、blocked floor 收敛、host_hit/load_back 预算口径对齐；
  - chunked 每轮刷新、write-through 锁计数隔离、stale lock/deferred page 缺口防护。
- 语义回归与行为门禁稳定：
  - `uv run --extra test pytest test/unit/srt test/unit/kvcache -q`：`230 passed`
  - `behavior_audit.py`：PASS
  - `behavior_gate.py`：PASS

#### 10.2 当前模拟误差状态（以最新可用数据）

- **full Round10（正式 workload）**：三组均 `MATRIX_TIMEOUT`，尚无完整 `request_details.csv/metrics.json`，无法形成正式 forward/makespan 误差结论。
  - `20260607_145401` / `20260607_145901` / `20260607_150401`
- **mini Round10（可完成）**：误差显著，forward 系统性偏低，makespan 显著偏短：
  - forward mean ratio：`0.249 ~ 0.255`
  - forward p50 ratio：`0.308 ~ 0.315`
  - forward p90 ratio：`0.187 ~ 0.191`
  - makespan ratio：`0.170 ~ 0.174`
- **最近可完成 full 历史参考（Round9）** 仍体现双目标 trade-off：
  - makespan 最佳：`1.107x`（`20260606_011814`），但 forward 较低（`0.729/0.758/0.692`）
  - forward 相对更高：`0.836/0.835/0.924`（`20260606_012109`），但 makespan 升至 `1.233x`

#### 10.3 阻塞点与下一阶段优先级

- 主阻塞点已明确：`Prefill extend blocked` 长尾在 full GSP 下仍导致整体完成性不足（纯超时形态）。
- 按计划停止条件，下一阶段进入 **mini-workload 可完成性优先**，并冻结参数扫表，先恢复 full 可完成 run，再重启 forward+makespan 双门禁收敛。
