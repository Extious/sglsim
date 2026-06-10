# Multi-Agent Fault Tolerance Benchmark 设计说明

本文档说明 SGLSim 中用于评估 GPU backup fault tolerance 的 multi-agent benchmark 构建方式。该 benchmark 面向 online 模式：启动真实 SGLang 风格进程拓扑和 HTTP 服务，由真实异步客户端发起请求，但 GPU forward 被 AIConfigurator 时延预测替代。实验目标是证明：在 multi-agent 场景中，只备份关键 agent 的 KV cache，可以在 GPU failure 发生时显著降低 recompute，并改善故障恢复后的服务指标。

## 目标

该 benchmark 主要回答四个问题：

1. GPU backup 相比 baseline 能否减少 failure 后的 KV recompute。
2. 只备份 critical agent 是否能保护关键请求，同时控制备份范围。
3. peer backup lag 对 KV reuse 和 recompute 的影响是什么。
4. failure 注入时间不同，failover overhead、retry 数量、吞吐和端到端延迟如何变化。

当前完整实验矩阵已经跑完 5 repeats，最终 artifact 位于：

```text
benchmark/synthesis_requests/results/ft_hpca_figures_final/
```

## 基础配置

主配置目录：

```text
benchmark/synthesis_requests/config/ma_ft_gpu_backup_critical/
```

baseline 配置目录：

```text
benchmark/synthesis_requests/config/ma_ft_baseline/
```

核心 client 配置如下：

```json
{
  "dataset_name": "generated-shared-prefix",
  "request_rate": "inf",
  "max_concurrency": 4,
  "gsp_num_groups": 4,
  "gsp_prompts_per_group": 1,
  "gsp_system_prompt_len": 4096,
  "gsp_question_len": 12000,
  "gsp_output_len": 384,
  "gsp_num_turns": 3,
  "gsp_ordered": true,
  "gsp_critical_agent_ids": ["1", "3"],
  "gsp_critical_agent_ratio": 0
}
```

该配置表示：

- 4 个 agent。
- 每个 agent 3 轮对话。
- 每轮目标输出 384 tokens。
- 每个 agent 第一轮包含长 system prompt 和 question。
- 后续 turn 延续同一个 chat session。
- 4 个 agent session 并发运行。
- agent `1` 和 agent `3` 被标记为 critical agent。

核心 server 配置：

```json
{
  "model_path": "Qwen/Qwen3-8B",
  "cache": {
    "enable_hierarchical_cache": false
  },
  "predictor": {
    "name": "aiconfigurator"
  },
  "scheduler": {
    "dp_size": 2,
    "load_balance_method": "round_robin"
  },
  "fault_tolerance": {
    "enabled": true,
    "strategy": "gpu_backup",
    "backup_policy": "critical_agents",
    "peer_backup_lag_tokens": 8,
    "backup_prefill_tokens": "all",
    "backup_decode": true,
    "trigger": {
      "enabled": true,
      "fault_dp_rank": 0,
      "inject_after_s": 30,
      "recover_after_s": 120
    }
  }
}
```

注意：`enable_hierarchical_cache=false` 是有意设计。当前 GPU backup 策略实现于 no-HiCache 链路，因为该链路的 simulator 准确度更稳定，也避免把 HiCache write-back/load-back 行为和 fault tolerance 机制混在一起。

## Workload 构建

workload 由 `benchmark/synthesis_requests/utils.py` 中的 `generate_shared_prefix_workload()` 生成。

生成步骤：

1. 根据 `gsp_num_groups` 生成多个 group。每个 group 对应一个 agent 的共享上下文来源。
2. 为每个 group 生成一个 system prompt。
3. 为每个 agent 的每一轮生成一个 question。
4. 如果 `gsp_num_turns > 1`，则每个 `WorkloadRequest.prompt` 是一个 prompt list。
5. 第 0 turn 的 prompt 是：

```text
system_prompt + "\n\n" + question_0
```

6. 后续 turn 的 prompt 是：

```text
question_1
question_2
...
```

因此，一个 agent session 不是单条独立请求，而是一个连续多轮对话。

## Multi-Agent Session 执行方式

client 执行逻辑位于 `benchmark/synthesis_requests/client.py`。

每个 agent 会被转换为一个 `PreparedRequest`：

```text
PreparedRequest(
  rid="1",
  turn_prompts=(turn0, turn1, turn2),
  agent_id="1",
  critical_agent=True
)
```

当 `turn_prompts` 存在且长度大于 1 时，client 走 multi-turn path：

```text
run_multi_turn_session_sync()
```

该函数为每个 agent 维护一个 `messages` 列表：

1. 发送 turn 0 到 `/v1/chat/completions`。
2. 收到 assistant output。
3. 把 assistant output 追加回 `messages`。
4. 发送 turn 1。
5. 重复直到最后一轮。

单个 agent 内部是顺序执行，因为后续 turn 依赖前一轮 assistant 输出。不同 agent 之间使用 `asyncio` 并发执行，并由 `max_concurrency` 控制最大并发 session 数。

当前配置中：

```text
agents = gsp_num_groups * gsp_prompts_per_group = 4
turns_per_agent = 3
HTTP generation requests = 4 * 3 = 12
max_concurrency = 4
```

也就是说，每次 run 中有 4 个 agent session 并发，每个 session 内部连续发 3 轮 chat completion。

## Critical Agent 标记

critical agent 有两种来源：

1. 显式配置：

```json
"gsp_critical_agent_ids": ["1", "3"]
```

2. sweep 中动态设置：

```json
"gsp_critical_agent_ratio": 0.75
```

workload 生成时会把二者合并为 `critical_ids`。每个 request 会携带：

```text
agent_id
critical_agent
```

发送给 runtime 的 HTTP payload 中还会加入 simulation metadata：

```json
{
  "simulation": {
    "parallel_worker_client": true,
    "worker_id": "1",
    "worker_seq": 0,
    "agent_id": "1",
    "critical_agent": true,
    "backup_kv": true
  }
}
```

字段含义：

- `parallel_worker_client`：标识该请求来自并发 multi-agent worker。
- `worker_id`：agent/session ID。
- `worker_seq`：该 agent 内部第几轮 turn。
- `agent_id`：用于统计和绘图的 agent ID。
- `critical_agent`：该 agent 是否关键。
- `backup_kv`：是否需要对该请求的 KV 做 GPU backup。

当前实现中，`backup_kv` 与 `critical_agent` 一致，即 critical agent 才触发 backup。

## Runtime 中的 GPU Backup 策略

runtime 逻辑位于 `python/sglsim/srt/managers/scheduler.py`。

GPU backup 是否启用由以下条件决定：

```text
fault_tolerance.enabled == true
strategy == "gpu_backup"
cache.hierarchical_enabled == false
```

也就是说，只有在 no-HiCache 路径上才启用本策略。

对单个 request，是否进行 backup 由 `_req_gpu_backup_enabled()` 判断：

```text
if backup_policy != "critical_agents":
    return True

return simulation.critical_agent or simulation.backup_kv
```

当 request 被判定需要 backup：

- prefill 后，如果 `backup_prefill_tokens == "all"`，则认为完整 prompt KV 已被 peer GPU 备份。
- decode 阶段，如果 `backup_decode == true`，则根据 `peer_backup_lag_tokens` 计算 durable decode tokens。
- failure snapshot 中记录 `pre_failover_backed_up_tokens`。
- failover retry 时，只有未被 backup 覆盖的 tokens 需要 recompute。

这对应论文叙事中的关键点：critical agent 的长上下文 KV 已在 failure 前被 peer backup，因此 failure 后不需要从零 recompute。

## Online Failure 注入

online fault injection 通过 HTTP endpoint 完成：

```text
POST /simulate_gpu_failure
POST /simulate_gpu_recovery
```

入口位于：

```text
python/sglsim/srt/entrypoints/http_server.py
```

触发链路：

```text
HTTP endpoint
  -> Engine.simulate_gpu_failure()
  -> TokenizerManager.simulate_gpu_failure()
  -> DataParallelController.handle_simulate_gpu_failure()
  -> Scheduler.handle_simulate_gpu_failure()
```

benchmark runner 会在指定时间点向 server 发送 failure 请求，例如：

```json
{
  "fault_dp_rank": 0,
  "inject_after_s": 40,
  "recover_after_s": 120
}
```

故障发生后，受影响请求会被记录为 failover retried，并保留：

- `failed_dp_rank`
- `pre_failover_output_tokens`
- `pre_failover_backed_up_tokens`
- `recompute_tokens`
- `failover_overhead_s`
- `agent_id`
- `critical_agent`

这些字段写入每个 run 的 `request_details.csv`，后续用于聚合图表。

## Sweep Matrix

完整 sweep 由：

```text
benchmark/synthesis_requests/diagnostics/run_fault_tolerance_sweeps.py
```

生成。

实验轴包括：

### 1. Strategy

比较 baseline 和 GPU backup：

```text
baseline
gpu_critical
```

该实验回答：GPU backup 是否减少 failure 后 recompute。

### 2. Backup Lag

扫描 peer backup 滞后：

```text
0, 1, 4, 8, 16, 32 tokens
```

该实验回答：decode backup 的滞后程度如何影响 KV reuse。

### 3. Critical-Agent Coverage

扫描 critical agent 覆盖比例：

```text
0.0, 0.25, 0.5, 0.75, 1.0
```

该实验回答：备份越多 critical agent，critical recompute 是否持续下降，以及 non-critical recompute 如何变化。

### 4. Failure Timing

扫描 failure 注入时间：

```text
30s, 36s, 40s, 44s
```

该实验回答：failure 发生在 prefill/decode/请求尾部附近时，failover overhead 和 retry 数量如何变化。

每个点使用 5 个 seeds：

```text
seed = seed_base + repeat_idx
```

最终总 run 数：

```text
cases = 2 + 6 + 5 + 4 = 17
repeats = 5
total = 17 * 5 = 85
```

最终 manifest 状态：

```json
{
  "total_runs": 85,
  "completed": 85,
  "failed": 0,
  "pending": 0,
  "axis_completed": {
    "strategy": 10,
    "backup_lag": 30,
    "critical_ratio": 25,
    "failure_timing_s": 20
  }
}
```

## 指标与图表

每个 run 产生：

```text
metrics.json
request_details.csv
simulator_config.json
```

plot 脚本读取所有 run 的 manifest 和 CSV，聚合出：

- recompute tokens
- reused tokens
- backed-up tokens
- KV reuse rate
- failover overhead
- output throughput
- mean e2e latency
- critical/non-critical agent latency
- requests retried

最终图表由以下脚本生成：

```text
benchmark/synthesis_requests/diagnostics/build_fault_tolerance_artifacts.py
```

输出目录：

```text
benchmark/synthesis_requests/results/ft_hpca_figures_final/
```

主要 artifact：

```text
fault_tolerance_sweep_hpca.png
fault_tolerance_sweep_hpca.pdf
fault_tolerance_sweep_system.png
fault_tolerance_sweep_system.pdf
fault_tolerance_paper_tables.md
fault_tolerance_paper_tables.tex
fault_tolerance_paper_tables.json
fault_tolerance_verification.json
```

## 最终 5-Repeat 结果

最终 paper verifier 已通过，关键结果如下：

```text
Recompute reduction: 57.5%
Critical-agent recompute reduction: 100.0%
GPU backup KV reuse: 20.0%
Output throughput change: +23.7%
Mean e2e latency change: -12.7%
Mean failover overhead change: -1.3%
```

这些结果说明：

1. baseline 在 failure 后需要重算大量 KV。
2. `gpu_backup + critical_agents` 能显著减少 recompute。
3. critical agent 的 recompute 几乎被完全消除。
4. GPU backup 没有带来明显端到端延迟恶化，在该 workload 下吞吐反而更好。
5. failure timing 越靠近请求尾部，retry 和 overhead 越低。

## 复现实验命令

运行完整 5-repeat sweep：

```bash
uv run --extra test python benchmark/synthesis_requests/diagnostics/run_fault_tolerance_sweeps.py \
  --resume \
  --repeats 5 \
  --seed-base 100 \
  --port-base 31000 \
  --results-root benchmark/synthesis_requests/results/ft_hpca_sweep \
  --run-timeout-s 900 \
  --log-level info
```

查看 sweep 状态：

```bash
uv run --extra test python benchmark/synthesis_requests/diagnostics/run_fault_tolerance_sweeps.py \
  --status \
  --repeats 5 \
  --seed-base 100 \
  --port-base 31000 \
  --results-root benchmark/synthesis_requests/results/ft_hpca_sweep
```

生成最终 artifact：

```bash
uv run --extra test python benchmark/synthesis_requests/diagnostics/build_fault_tolerance_artifacts.py \
  benchmark/synthesis_requests/results/ft_hpca_sweep/fault_tolerance_sweep_latest.json \
  --output-dir benchmark/synthesis_requests/results/ft_hpca_figures_final
```

运行 paper verifier：

```bash
uv run --extra test python benchmark/synthesis_requests/diagnostics/verify_fault_tolerance_artifacts.py \
  benchmark/synthesis_requests/results/ft_hpca_figures_final/fault_tolerance_artifacts.json \
  --profile paper
```

## 验证

已通过的相关测试：

```text
uv run --extra test pytest test/unit/benchmark test/unit/srt/test_online_fault_tolerance.py test/unit/core/test_import_boundaries.py -q
40 passed
```

```text
uv run --extra test pytest test/integration -q
4 passed
```

全量测试结果：

```text
uv run --extra test pytest test/unit test/integration -q
329 passed, 6 failed
```

其中 6 个失败位于 HiCache/KV cache accounting 单测，和本文档描述的 no-HiCache `gpu_backup` fault tolerance 路径不同。

## 论文写作建议

可以把该 benchmark 描述为：

> We construct an online multi-agent workload where each agent maintains a multi-turn chat session with long shared context. Agents execute concurrently through the real HTTP serving path. A configurable subset of agents is marked critical, and only their KV cache is backed up to peer GPUs. We inject logical GPU failures at runtime and compare baseline retry with GPU-backed failover in terms of recomputation, KV reuse, latency, throughput, and critical-agent protection.

中文版本：

> 我们构建了一个 online multi-agent workload。每个 agent 都维护一个多轮 chat session，并带有长共享上下文；多个 agent 通过真实 HTTP serving path 并发执行。实验中可配置一部分 agent 为 critical agent，系统只对这些关键 agent 的 KV cache 进行 peer GPU backup。随后在运行时注入逻辑 GPU failure，并从 recompute tokens、KV reuse、latency、throughput 和 critical-agent protection 等维度比较 baseline retry 与 GPU-backed failover。

该表述突出了三个贡献点：

1. benchmark 是 online serving，而不是 offline trace replay。
2. workload 是 multi-agent multi-turn，而不是简单批量独立请求。
3. fault tolerance 策略是 critical-agent-aware GPU KV backup，而不是全量备份。
