# Multi-Agent AegisKV 容错 Benchmark（三策略 v2）

本 benchmark 在两轮 fan-out/fan-in coding workflow 上对比三种 **GPUReplica** 备份策略：

| 策略 | 行为 |
|------|------|
| `baseline` | 不备份 KV |
| `naive_backup` | 备份所有 eligible segment 的 KV |
| `aegiskv` | 仅备份 **Critical Agent**（O0 / S2 / S5）的 KV |

**Agent 拓扑**：主 agent `O0` + 6 个 subagent `S1`–`S6`（每轮 3 个并行）。`S2`、`S5` 为 heavy-decode critical subagent。主路径为 **offline policy simulator**，产出 per-workflow WCT 与 per-agent 时间线。

## Workflow

```text
O0 -> S1/S2/S3 -> O0 -> S4/S5/S6 -> O0
```

KV segment 使用 `P1`–`P8`（O0 各阶段）与 `S10`–`S24`（subagent 分支）等局部 ID，与 agent ID 区分。

## 实验组

- `strategy`：三策略 × S5_ToolBlocked（主结果）
- `failure_site`：三策略 × 6 个 failure site
- `scale`：三策略 × 并发 workflow {2,4,8,16}

## 命令

Smoke（model 模式）：

```bash
uv run --extra test python benchmark/fault_tolerance/multi_agent/run.py \
  --config-dir benchmark/fault_tolerance/multi_agent/config/aegiskv_coding \
  --output-dir /tmp/aegiskv_v2_smoke \
  --mode model --smoke
```

完整 paper run：

```bash
uv run --extra test python benchmark/fault_tolerance/multi_agent/run.py \
  --config-dir benchmark/fault_tolerance/multi_agent/config/aegiskv_coding \
  --output-dir benchmark/fault_tolerance/multi_agent/results/aegiskv_v2 \
  --mode model --repeats 5 \
  --experiments strategy,failure_site,scale
```

出图：

```bash
uv run --extra test python benchmark/fault_tolerance/multi_agent/plot.py \
  --results-dir benchmark/fault_tolerance/multi_agent/results/aegiskv_v2
```

论文表与验证：

```bash
uv run --extra test python benchmark/fault_tolerance/multi_agent/paper_tables.py \
  --results-dir benchmark/fault_tolerance/multi_agent/results/aegiskv_v2
uv run --extra test python benchmark/fault_tolerance/multi_agent/verify_artifacts.py \
  --results-dir benchmark/fault_tolerance/multi_agent/results/aegiskv_v2 --profile paper
```

## 输出

- `runs.csv` / `segments.csv` / `recovery_paths.csv`
- **`workflows.csv`**：每个 workflow 的 `wct_s`、`impacted_by_failure`
- **`agent_timelines.csv`**：每个 agent 的 `start_s`/`end_s`/`recovery_extra_s`/`is_failure_target`
- 主图：`fig_wct_cdf`、`fig_agent_timeline`、`fig_agent_timeline_compare`、`fig_agent_duration_violin`
- 辅图：`fig_strategy_recovery`、`fig_backup_cost`、`fig_recovery_vs_cost`、`fig_scale`

## 论文解读

- **WCT CDF**：AegisKV 与 NaiveBackup 曲线左移且接近，显著优于 Baseline tail
- **Agent 时间线**：failure 仅拉长 S5（或对应 site）的 recovery 段；Aegis 与 Naive 在该段接近
- **Backup 成本**：AegisKV backed-up tokens 远低于 NaiveBackup，买到相近 WCT
