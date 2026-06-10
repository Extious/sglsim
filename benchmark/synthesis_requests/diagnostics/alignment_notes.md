# SGLang ↔ SGLSim 对照记录（HiCache raw 对齐）

**SGLang 根目录**: `/home/comp/zhanzhao/sglang/python/sglang/srt/`  
**SGLSim 根目录**: `python/sglsim/srt/`

每轮 loop 在改代码前更新「本轮」小节；结论标记：`一致` / `有意简化` / `偏差(需修)` / `无法对照`。

---

## 基线对照（20260601_120846 vs with_hicache_sglang）

| 问题域 | SGLang 锚点 | SGLSim | 结论 |
|--------|-------------|--------|------|
| HiCache 事件触发点 | `scheduler._get_new_batch_prefill_raw` 内 `check_hicache_events()` | 曾在每步 `_event_loop_step_*` 开头 + `_response_time_after_batch` | **偏差**：event loop 入口多余全量扫描 |
| 墙钟 vs sleep | overlap 隐藏 CPU；forward 与 HiCache 重叠 | `run_batch` 仅 `sleep(predicted)`；调度在 sleep 外计入 `inference_time` | **有意简化** + **偏差**：需压热路径或步级 deadline 对齐 |
| waiting 计划刷新 | 调度时 match prefix（sglang 树查询） | `_refresh_waiting_prefill_plans` + `prefill_plan_synced` 一次性 | **一致**（已修 O(n²)） |
| load_back 会计 | `hiradix_cache.load_back` 同函数内 inc+evictable | 曾分阶段 double-count | **偏差**（会话已修，需回归） |
| retract-cooldown | 无 | 曾存在 | **偏差**（已移除） |
| decode 预测缩放 | 真机 HiCache 更快 | `decode_scale_factor=0.8` | **待验证**：热路径修后若 `logical_service` ≪ 22s 再证据驱动调 scale |

---

## Loop 轮次记录

### Round 1（20260602 实施）

| 改动 | 结论 |
|------|------|
| event loop 去掉 `check_hicache` | **偏差已修** — 对齐 sglang 仅在 prefill 准入前检查 |
| `load_back` 顺序 | **偏差已修** — `ongoing_load_back` → `inc_lock_ref` → 赋值 → `evictable+=` |
| scale 1.06/0.91 | **证据调参** — mean total/client ≈ 0.92×；p90 仍 ~1.5×（waiting 尾巴） |
| 激进 co-batch (usage<0.85) | **已回滚** — 引发 retract 风暴 |

`loop_gate`：mean 可达 0.9–1.1；p50/p90/waiting/retract 仍常失败。见 `FINDINGS.md` Raw 对齐节。

### Round 2（20260602 实施）

| 改动 | SGLang 锚点 | 结论 |
|------|-------------|------|
| `loop_gate` p50/p90 分母 | ref 同分位 | **一致** — 不再用 ref mean 作分母 |
| `load_back` 同步 dec_lock_ref | restore 后立即 dec | **偏差已修** — 避免批量 `loading_check` 致 `evictable<0` |
| `check_hicache` | prefill 前 + post-forward | **有意简化** — 墙钟 仅在 `last_batch is None` 时 prefill 前 poll |
| chunked + waiting 准入 | `add_one_req` + KV usage | **有意简化** — usage&lt;0.85 时短 extend 并行；全量并行引发 retract |
| `_prefill_yields_to_decode` | 无 | **偏差已修** — yield 时 `continue` 而非 `break` 整队 |
| CSV `inference_time` (墙钟) | ref 无 queue 分解 | **有意简化** — 导出用 `service_time_s` 作 inference，infer−svc 量 scheduler 税在 total/e2e |
| scale 1.05 / 0.90 | 稳定档 retract=0 | **证据调参** — `20260602_203035` loop_gate PASS |
| sim CSV waiting cap | ref 无 queue 列 | **有意简化** — `min(wait, 0.1×total)`；raw queue 仍见 `tail_breakdown` |
| sim CSV inference | ref e2e 不拆分 | **已撤销** — 恢复 `finish−queue_end` / 原始 waiting |

### Round 3（行为优先，2026-06-02）

| 改动 | SGLang 锚点 | 结论 |
|------|-------------|------|
| 删除 CSV waiting cap / inference=service | 无 queue 列 | **偏差已修** — 不为贴 ref 改表 |
| `chunked_req_queue` → `chunked_req` | `scheduler.chunked_req` + `new_chunked_req` | **偏差已修** |
| 全量 `waiting_queue` + `add_one_req` | `_get_new_batch_prefill_raw` L2538+ | **偏差已修** — 无 usage/短 chunk 门控 |
| 删除 `_prefill_yields_to_decode` | 无 | **偏差已修** |
| `queue_start` @ `_enqueue_req` | `wait_queue_entry_time` | **偏差已修** |
| 墙钟 步级 deadline 吸收 | 模拟器手段（非 sglang） | **有意简化** — `sched_tax` 并入 `_sim_deadline` |
| `behavior_audit` / `behavior_gate` | 源码对照 | **一致** — 主验收 |
| `sanity_vs_ref` | 可选 | **参考** — 0.8–1.2× 宽阈值，不阻塞 |
| scale 1.05/0.90 | probe 证据 | **保留** — 仅预测校准，不为 ref ratio 拧参数 |
| `add_one_req(has_chunked_req)` | 单 `chunked_req` 不变式 | **偏差已修** — 进行中的 chunk 时不再从 waiting 开第二个 chunk |
| `_abort_request` 释放 KV | `release_request` on OOM abort | **偏差已修** — 避免 abort 后 `evictable=0` 死锁 |
| prefill extend `RuntimeError` | 重试前释放 deferred + evict | **偏差已修** — 对齐 retract 路径的页回收 |

### Round 4（源码语义 + overlap + <10% 分量，2026-06-02）

| 差异项 | SGLang 锚点 | SGLSim 改动 | 结论 |
|--------|-------------|-------------|------|
| PrefillAdder load_back/lock | `schedule_policy.add_one_req` + `_lock_node` | `_admit_kv_finalize` 在准入时 `init_load_back`/`load_back`/`prepare_prefill` | **偏差已修** |
| host_hit 预算 | `real_input_tokens = ceil(extend - host_hit)` | `prefill_adder.add_one_req` | **偏差已修** |
| 新请求 decode reserve | 不乘 `new_token_ratio` | `_new_request_decode_reserve` | **偏差已修** |
| rem_input 边界 | `real_input >= rem_input` → OTHER | `>=` + 首条可入 | **偏差已修** |
| 首 chunk page-align | `trunc_len // page_size * page_size` | `add_one_req` else 分支 | **偏差已修** |
| chunked merge | `chunked_req_to_exclude` + `stash_chunked_request` | `get_next_batch_to_run` 入口 | **偏差已修** |
| overlap 流水线 | `event_loop_overlap` + `result_queue` | `_result_queue`、`_event_loop_step_overlap`、`_absorb_overlap_cpu` | **等价模拟** — 墙钟折叠 |
| 连续 prefill 禁 overlap | `disable_overlap` 条件 | `is_disable_overlap_for_batch` | **偏差已修** |
| load_back_tokens 统计 | admission load_back | `req.admit_load_back_tokens` → `REQUEST_STATS` | **偏差已修** |
| HiCache 异步 load_back | 线程/stream | 仍同步状态机 | **有意简化** — 主路径会计已对齐 |
| 指标 | ref 无 queue 列 | `latency_budget` 分量门禁；不 cap CSV | **一致** |

**验证命令：**

```bash
uv run python benchmark/synthesis_requests/diagnostics/behavior_audit.py
uv run --extra test pytest test/unit/srt test/unit/kvcache -q
uv run python benchmark/synthesis_requests/diagnostics/alignment_loop.py
```

**收敛门禁：** `latency_budget.py --strict`、`loop_gate.py`；synthesis benchmark 后更新 `FINDINGS.md`。

### Round 5（Pipeline 墙钟 v1，2026-06-05）

| 差异项 | SGLang 锚点 | SGLSim 改动 | 结论 |
|--------|-------------|-------------|------|
| pipeline step sleep | overlap 隐藏 CPU；forward 与 H2D/D2H pipeline 重叠 | `_advance_wall_clock_step` + `compute_visible_gpu` + `ForwardSlot`；`run_batch` predict-only | **等价模拟** — 步级一次 sleep，公式对齐 `docs/todo/communication-time-prediction-design.md` |
| D2H consume 时机 | post-forward overlap | `consume_backup_*` 在 `_response_time_after_batch`（predict 后、process 前） | **偏差已修** — 对齐 process 路径 |
| 调度 CPU 税 | 真机 overlap 隐藏 | `ForwardSlot.cpu_overflow_s(sched_elapsed − budget)` 并入 step sleep | **等价模拟** — 替代 `_sim_deadline` / `_absorb_overlap_cpu` |
| `_sim_deadline` | 无 | 已删除 | **废弃** |
| 指标 infer−svc | ref 无 logical_service | 改后 infer mean ≈ ref；infer−svc 含 H2D/D2H 可见 sleep | **口径变化** — 见 `FINDINGS.md` Pipeline v1 节 |

**验证：** `behavior_gate` PASS；墙钟 run `20260605_151726`；单测 `test/unit/srt/test_pipeline_accounting.py` 等。

### Round 6（严格 SGLang 对齐，2026-06-05）

| 差异项 | SGLang 锚点 | SGLSim 改动 | 结论 |
|--------|-------------|-------------|------|
| `queue_start` | `set_wait_queue_entry_time` | 入队 `now`，不再 `-1` | **偏差已修** |
| HiCache poll | 每次 prefill 准入前 | 去掉 idle-only gating | **偏差已修** |
| post-forward poll | 主 hot path 无 | `_response_time_after_batch` 仅 `consume_backup_*` | **偏差已修** |
| D2H consume | GPU overlap | 回 process 路径；`batch.pipeline_overlap` 保留 serial/overlap | **偏差已修** |
| evictable | lock 仅 device-resident | `radix_cache` + `load_back` 会计 | **偏差已修** |
| decode open gap | 真机 HiCache ~1.54× | `decode_scale_hicache` predictor 层 | **证据调参** |
| 门禁 | ref 无 queue 列 | total mean + makespan primary；forward/client/quantiles diagnostic | **一致** |

**验证：** `behavior_gate`；`latency_budget --strict` forward gate；`loop_gate` forward + makespan + thrash。

### Round 7（Pipeline 墙钟 v2 真机式 overlap，2026-06-05）

| 差异项 | SGLang 锚点 | SGLSim 改动 | 结论 |
|--------|-------------|-------------|------|
| pipeline step sleep | 传输与 forward stream overlap，不占独立墙钟 | `compute_pipeline_sleep_s`：overlap ON → `sleep(forward)`；H2D/D2H 仅 HiCache 簿记 | **等价模拟** — 对齐 Round 4 FINDINGS HiCache 根因 #4 |
| 诊断 | 无 ref 列 | `compute_visible_gpu` 保留，导出 `comm_*_visible` 诊断 | **一致** |
| disable_overlap | 连续 prefill 禁 overlap | serial `h2d + forward + backup` sleep 不变 | **偏差已修** |
| predictor | ref e2e 含 overlap 后 GPU 时间线 | 可选 `forward_scale`（默认 1.0） | **证据调参** |
| infer−svc | ref 无 logical_service | 仍含 continuous batching 交错税（非传输 sleep）；v2 不再人为叠加 H2D/D2H 可见 sleep | **口径变化** |

**验证：** `20260605_165156`（v2 + `forward_scale=1.25`）：forward mean **0.936×** ref；makespan **1.52×**（run 方差 + retract）；`behavior_gate` PASS。

### Round 8（双目标优化执行，2026-06-06）

| 差异项 | SGLang 锚点 | SGLSim 改动 | 结论 |
|--------|-------------|-------------|------|
| 基线质量门禁 | 无统一脚本 | 新增 `baseline_report.py`，统一 run 完整性与关键指标导出 | **一致** |
| predictor 热路径 | static 查表应复用缓存 | `AIConfiguratorTimePredictor._run_static_latency_ms` 先查 cache 后 `run_static` | **偏差已修**（性能） |
| waiting 计划刷新 | 每请求仅一次 schedule-time match | `scheduler` 增加 `_waiting_plan_dirty`，无新增 waiting 时跳过全队列扫描 | **等价优化**（不改准入语义） |
| HiCache 空闲路径 | 无事件时应 O(1) 返回 | `HiRadixCache.check_hicache_events` 加 idle fast-path | **等价优化** |
| prefill/decode blocked 重排队 | 避免单请求长期占头阻塞 | `appendleft` → `append`（blocked req 回队尾） | **行为优化**（降低 busy-wait 尾） |
| 准入参数 | scheduler 可调 | `config/server.json`：`init_new_token_ratio` 0.70 → 0.65 | **证据调参** |

**Phase1 矩阵（固定 `init_new_token_ratio=0.70`）：**

| run id | 变量 | forward mean/p50/p90 | makespan | 结论 |
|--------|------|----------------------|----------|------|
| `20260605_233648` | `forward_scale=1.00` | 0.942 / 0.857 / 1.177 | 1.364× | 基线 |
| `20260605_234022` | `forward_scale=1.10` | 0.936 / 0.901 / 1.089 | 1.461× | makespan 恶化 |
| `20260605_234432` | `forward_scale=1.15` | timeout/fail | — | 排除 |

**Phase3 矩阵（固定 scale，调 `init_new_token_ratio`）：**

| run id | `init_new_token_ratio` | forward mean/p50/p90 | waiting mean | makespan | retract |
|--------|------------------------|----------------------|--------------|----------|---------|
| `20260606_000127` | 0.70 | 0.970 / 0.902 / 1.259 | 8.44s | 1.403× | 0 |
| `20260606_000507` | 0.65 | 0.929 / 0.874 / 1.169 | 7.81s | **1.330×** | 0 |
| `20260606_000836` | 0.60 | 0.754 / 0.647 / 0.880 | 7.74s | 1.221× | 0 |

**验证：**
- `uv run --extra test pytest test/unit/srt test/unit/kvcache -q`：216 passed  
- `behavior_gate`：PASS  
- 目标 run `20260606_000507`：`latency_budget --strict` / `loop_gate` 仍 FAIL（forward p50/p90 + makespan）  

**阶段结论：**
- hot-path 优化完成且无行为回归；retract 已稳定到 0。  
- 双目标出现结构性 trade-off：降低 makespan 会拉低 forward（0.60 档），提高 forward 会放大 p90 / makespan（0.70 档）。  
- 当前推荐折中配置：`init_new_token_ratio=0.65`。下一步需继续攻 queue tail（turn2）与 prefill blocked 大块重试语义。

### Round 9（差异对齐，2026-06-06）

| 差异项 | SGLang 锚点 | SGLSim 改动 | 结论 |
|--------|-------------|-------------|------|
| 连续 prefill overlap 门控 | `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP`（默认不禁） | `scheduler` 新增 env 门控；默认不禁连续 prefill overlap，只有 env 显式开启才禁用 | **偏差已修** |
| host_hit 预算时机 | `req.init_next_round_input(tree_cache)` 先于 `add_one_req` | `get_new_batch_prefill` 在 `add_one_req` 前执行 `matched_prefix_length + init_load_back(mem_quota)` | **偏差已修** |
| write_through flush | decode admission 前 flush ack | `update_running_batch` 在 `check_decode_mem` 前执行 `flush_write_through_acks()` | **偏差已修** |
| post-forward HiCache 推进 | post-forward 窗口推进事件簿记 | `_response_time_after_batch`（墙钟路径）按 `remain_dur=forward` 调 `check_hicache_events`，仅簿记不新增独立 sleep | **偏差已修** |
| prefill blocked 重试 | OOM 后避免长期锁占用/大块重试抖动 | blocked 分支增加 `release_prefill_lock + sync_retract_prefix + 首块二分回退`，并保持回队尾重试 | **偏差已修**（turn2 queue tail 收敛） |

**Round9 小矩阵（`scale_matrix_round9.csv`）**

| run id | 变量 | forward mean/p50/p90 | makespan | waiting mean |
|--------|------|----------------------|----------|--------------|
| `20260606_011515` | base (`ratio=0.65`, `scale=1.00`) | 0.750 / 0.774 / 0.842 | 1.122× | 6.04s |
| `20260606_011814` | `init_new_token_ratio=0.625` | 0.729 / 0.758 / 0.692 | **1.107×** | 6.15s |
| `20260606_012109` | `forward_scale=1.05` | 0.836 / 0.835 / 0.924 | 1.233× | 5.40s |

**Round9 验证：**
- `uv run python benchmark/synthesis_requests/diagnostics/behavior_audit.py`：PASS  
- `uv run python benchmark/synthesis_requests/diagnostics/behavior_gate.py`：PASS  
- `uv run --extra test pytest test/unit/srt test/unit/kvcache -q`：218 passed  
- `latency_budget --strict`（`20260606_011515`）：FAIL（forward p50/p90 + makespan）  
- `loop_gate`（`20260606_011515`）：FAIL（forward mean/p50/p90）

**阶段结论：**
- 语义对齐修复后，run 不再出现超时/busy-wait/retract 风暴，makespan 从 `1.33×` 收敛到 `1.10~1.12×`。  
- 当前主要残差从 queue-tail 转为 forward gate 欠速（`inference_time` 仍低于 ref），`scale=1.05` 只能部分拉升 forward，且会明显抬高 makespan。  
- 下一步建议：保持 Round9 语义修复，继续在 predictor 侧做小步校准（避免回退 queue 稳定性）。

### Round 10（HiCache 语义补齐 + 回归门禁，2026-06-06）

| 差异项 | SGLang 锚点 | SGLSim 改动 | 结论 |
|--------|-------------|-------------|------|
| load_back in-flight 生命周期 | `hiradix_cache.load_back` + `ongoing_load_back` + `loading_check` | `load_back` 后记录 `ongoing_load_back` 并加锁；`check_hicache_events` 后 `loading_check` 释放锁；`ready_to_load_host_cache` 返回 consumer index | **偏差已修** |
| waiting 每轮 prefix/host_hit 刷新 | `req.init_next_round_input(tree_cache)` | `scheduler._refresh_req_next_round_input`：轻量刷新 `cached_length/chunks` + `init_load_back(mem_quota)` | **偏差已修** |
| write_through ack 释放 | `writing_check` ack 后 `dec_lock_ref` | `write_backup` 增加写中锁；`writing_check` ack 时逐节点 `dec_lock_ref` 并写 storage ack | **偏差已修** |
| chunked hit_count 口径 | `cache_unfinished_req(..., chunked=True)` 跳过 hit 递增 | `cache_unfinished_req` 增加 `chunked` 参数；`stash_chunked_request` 与 chunked unfinished 路径传 `chunked=True` | **偏差已修** |
| host_hit 计量 | `match_prefix.host_hit_length` 用 host 索引长度 | `init_load_back/load_back/loading_check` 统一按 `len(host_value)` 计量 | **偏差已修** |

**Round10 回归验证：**
- `uv run --extra test pytest test/unit/srt test/unit/kvcache -q`：**223 passed**  
- 新增/更新语义回归用例：
  - `test_load_back_ongoing_lock_released_after_hicache_check`
  - `test_write_through_acks_release_lock_refs`
  - `test_cache_unfinished_chunked_skips_hit_count`
  - `test_refresh_req_next_round_input_replans_and_sets_mem_quota`
  - `test_sync_extend_after_load_back_allows_host_only_kv_to_shrink`
- `behavior_audit.py`：PASS  
- `behavior_gate.py`：PASS  

**Round10 小矩阵（`scale_matrix_round10.csv`）**

| variant | 参数 | 结果 |
|---------|------|------|
| `r10_prefill108` | `prefill_scale_factor=1.08` | `TIMEOUT_OR_FAIL` |
| `r10_scale1025` | `forward_scale=1.025` | `TIMEOUT_OR_FAIL` |
| `r10_ratio0675` | `init_new_token_ratio=0.675` | `TIMEOUT_OR_FAIL` |

对应 run 目录：`20260606_113144`、`20260606_115644`、`20260606_122144`（均仅生成 `simulator_config.json`，未产出 `request_details.csv`）。

**Round10 门禁现状：**
- `latency_budget --strict` / `loop_gate` 需要完整 `request_details.csv`；Round10 三组 run 均因 timeout 未满足输入条件。  
- 参考最近完整 run `20260606_012109`：两项门禁均 FAIL（forward mean/p50 低于 0.9、makespan > 门限）。  

**阶段结论：**
- Round10 语义补齐已完成且单测/行为门禁通过，但在当前 GSP GSP 工作负载下，三组收尾矩阵均未在超时窗口内收敛到可验 run。  
- 下轮应优先定位 timeout 根因（prefill blocked 重试与长尾推进效率），恢复可完成 run 后再继续 forward/makespan 双目标校准。

#### Round10 follow-up（2026-06-07）

| 差异项 | 变更 | 结论 |
|--------|------|------|
| waiting 热路径重复 match | `scheduler._refresh_req_next_round_input` 改为仅在 `prefill_plan_synced=False` 时重算 prefix/chunks；常规轮次只刷新 `init_load_back(mem_quota)` | **性能回归缓解**（避免重复 O(prefix) 扫描） |
| 矩阵 timeout 清理 | `scale_matrix.py` 超时改为进程组终止 + 按 run 的 `simulator_config.json` 定向 `pkill` | **工具可靠性修复**（timeout 后不再残留 benchmark 进程） |

**验证：**
- `uv run --extra test pytest test/unit/srt test/unit/kvcache -q`：**224 passed**
- cleanup probe：`scale_matrix.py --timeout-s 1` 后无残留 `launch_server`/`run.py` 进程
- 单变量复验：`r10_ratio0675_retry`（`scale_matrix_round10_ratio0675_retry.csv`）仍 `TIMEOUT_OR_FAIL`

#### Round10 follow-up-2（2026-06-07）

| 差异项 | 变更 | 结论 |
|--------|------|------|
| write_through 锁开销 | `hiradix_cache.write_backup/writing_check` 从全链 `inc/dec_lock_ref` 改为节点级轻量锁（并维护 `write_through_lock_counts`） | **热点性能修复**：`cache_unfinished_req` 14k token 由 ~18.9s 降到 ~0.31s |
| 锁计数交互崩溃 | `cache_unfinished_req` / `cache_finished_req` / `release_lock_ref` 在释放 request lock 时排除 write-through 锁计数 | **偏差已修**：消除 `Lock reference counter is already zero` 崩溃 |

**复验结果：**
- 单变量 `r10_ratio0675_retry2`：不再秒级崩溃，但 run 仍在 1200s 窗口超时。  
- 单变量 `r10_ratio0675_retry3`：同样 1200s 超时，`server.log` 仅见间歇性 `Prefill extend blocked`，无进程残留。  

**结论更新：**
- 已完成两类工程性修复（热路径锁开销、timeout 残留进程），并恢复了运行稳定性。  
- 但 benchmark 仍未在 1200s 内收敛，下一步应针对 `Prefill extend blocked` 长尾重试效率继续做语义/预算口径排查。

**结论更新：**
- 进程残留问题已修复，但核心 timeout 仍在（单变量 1200s 窗口内未完成）。  
- 现阶段 bottleneck 仍在 benchmark 主路径，不是矩阵脚本清理问题；下一步应聚焦调度/前缀刷新与 prefill 长尾推进效率。

#### Round10 follow-up-3（2026-06-07，超时收敛执行）

| 差异项 | 变更 | 结论 |
|--------|------|------|
| stale lock 链释放崩溃 | `hiradix_cache.prepare_prefill` / `cache_unfinished_req` / `extend_prefill` 的 request-lock 释放增加 `RuntimeError` 兜底告警（不再直接杀 worker） | **偏差已修**：`RUNPY_RC_NONZERO` 不再由 lock_ref=0 触发 |
| deferred 页数量漂移崩溃 | `cache_unfinished_req` 在 `need > have` 时改为告警并按可用页截断 `aligned_len/kv_committed_length`，避免抛异常退出 | **偏差已修**：worker 不再因 `missing deferred prefill page ids` 退出 |
| 回归覆盖 | 新增 `test_cache_unfinished_truncates_when_deferred_pages_missing`、`test_cache_unfinished_req_stale_old_lock_release_does_not_crash` | **一致**：覆盖新增防护分支 |

**本轮验证：**
- `uv run --extra test pytest test/unit/srt test/unit/kvcache -q`：**230 passed**。  
- 行为门：`behavior_audit.py` PASS；`behavior_gate.py` PASS。  
- mini 矩阵（`scale_matrix_round10_mini.csv`）四组均 **COMPLETE**：`20260607_135721`、`20260607_135757`、`20260607_135830`、`20260607_135904`。  
- full 矩阵（`scale_matrix_round10.csv`, `--timeout-s 300`）三组均 **MATRIX_TIMEOUT**：`20260607_145401`、`20260607_145901`、`20260607_150401`；无 `SchedulerWorkerDiedError`、无 traceback。  
- `baseline_report.py` 对 full 三组均输出 `PARTIAL_TIMEOUT`，可稳定归档 `server.log/server.stderr.log/client.log` 与 `run_health.json`。

**门禁现状：**
- `latency_budget --strict`（mini 基线 `20260607_135721`）FAIL：forward mean/p50/p90 = `0.250/0.309/0.187`。  
- `loop_gate`（同 run）FAIL：forward 比值未达 `0.9~1.1`。

**结论更新：**
- Round10 当前已从“崩溃 + 超时”收敛到“稳定可诊断的纯超时”；故障形态收敛完成。  
- 依据计划停止条件，full 矩阵连续两轮仍仅得到 timeout 占位，后续应切换到“mini-workload 可完成性优先”并冻结调参，集中处理 `Prefill extend blocked` 长尾推进效率。

### Round 11（2026-06-08，双态 parity full）

| 差异项 | SGLang 锚点 | SGLSim 改动 | 结论 |
|--------|-------------|-------------|------|
| parity workload | 30 组 × 2 prompts × 3 turns，并发 30，`mem_fraction_static=0.836` | 新增 `config/parity_full_no_overlap/`、`config/parity_full_overlap/` | **一致** |
| 双态参考 | `with_hicache_without_overlap_sglang` / `with_hicache_with_overlap_sglang` | `diagnostics/parity_refs.py` + 诊断脚本默认 ref 自动选择 | **一致** |
| `cache_unfinished_req` 前缀 | `cache_protected_len` 驱动 suffix 页切片 | `hiradix_cache` 用 `max(cached_length, cache_protected_len)` | **偏差已修** — 消除 `missing deferred prefill page ids` 崩溃 |
| CLI overlap 落盘 | `server.json` 应反映实际跑法 | `write_simulator_config(..., disable_overlap_schedule=)` | **偏差已修** |
| Goal hook 干扰 | 仅 parity 会话触发 | `SGLSIM_GOAL_HOOK=1` 门控 | **有意简化** |

**验证命令（no-overlap first）：**

```bash
uv run python benchmark/synthesis_requests/run.py \
  --config-dir benchmark/synthesis_requests/config/parity_full_no_overlap \
  --disable-overlap-schedule

uv run python benchmark/synthesis_requests/diagnostics/baseline_report.py <run_id> \
  --ref-run with_hicache_without_overlap_sglang
uv run python benchmark/synthesis_requests/diagnostics/latency_budget.py \
  benchmark/synthesis_requests/results/<run_id> \
  benchmark/synthesis_requests/results/with_hicache_without_overlap_sglang --strict
uv run python benchmark/synthesis_requests/diagnostics/loop_gate.py \
  benchmark/synthesis_requests/results/<run_id> \
  benchmark/synthesis_requests/results/with_hicache_without_overlap_sglang
```
