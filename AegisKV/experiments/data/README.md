# Workload data

论文 profile 使用 TraceLab `v0.0.2` 的 DuckDB release，文件不提交到仓库：

```bash
uv run --extra benchmark python AegisKV/experiments/run.py fetch-data
```

下载器要求 SHA-256 为
`a7bab286bc640844560850965ccf47975cf66407154132abaab90f27ec9be744`，并写入
`data/cache/syfi_coding_trace-v0.0.2.duckdb`。数据库在 2026-08-29 本地审计时包含
665,453 rounds、8,058 provider-session 和 52 个匿名用户；这些是 release artifact 的观测值，
不是从论文摘要复制的计数。

## P11 observed Weka/AgentX DAG

P11 的不可变编译产物位于
[`observed/weka_agentx_causal_v1`](observed/weka_agentx_causal_v1)。生成与审计命令为：

```bash
uv run python -m AegisKV.experiments.src.build_weka_observed_workload
uv run python -m AegisKV.experiments.src.build_weka_observed_workload --audit-only
```

源文件固定为 1,847,151,435-byte Weka materialized JSONL，SHA-256 为
`29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276`。393 个 workflow
中有 126 个包含 observed parallel stage；按 requests/workflow 的实际排序精确分为 42/42/42：
small=`30--162`、medium=`163--470`、long=`475--4114`。每个 stratum 为 10 个 seed
各选择 1 个 workflow，跨 seed 严格无放回；最终 30 个 workflow、13,918 个请求。

冻结 identity：

| 对象 | SHA-256 |
| --- | --- |
| `manifest.json` 文件 | `9e6ca27ebc70107836bc39f8d5ed945de9af5e2224a3ab6cfbaa21d1926f0eeb` |
| `requests.jsonl` | `7f0a9b18aa30dbf03dd349e15fc9f10f9d57f9acb041192d5a55404a0248551b` |
| request universe | `587278b60852e84fd258172ae9c9458d498f1316e502e68820b2bea80a54b9ae` |
| predictor lineage | `0e6c9acf78c644ee554ec8dcc9f15f07b5e07c97b79be31997eda37a01039dc3` |

预测器使用全部 126 个 eligible workflow 的 workflow-disjoint 5-fold cross-fit；请求 payload
不包含 actual remaining time/actual Slack。6,047 个选中请求有 admission-time modeled ticket，
7,871 个显式 fail-close-zero，即 modeled fraction 为 `0.434473`（43.45%）。当前 95%
workflow-simultaneous interval 的 observed coverage 为 `0.928571 < 0.95`，且 selected safe-Slack LCB
非零数为 0；这是必须报告的保守退化结果，不能称为目标达标、硬件校准或 paper-ready。Weka block
IDs 是 workflow-local，token identity 因而同时携带 workflow namespace，禁止虚假的跨 workflow
prefix 命中。

每个 P11 cell 只能读取其 `(seed, dag_stratum)` 对应的 **一个完整 workflow**；smoke 限制也不得切掉
workflow 中间的 request 后再声称 WCT。这个 artifact 只是 P11 的 observed-workload 输入，不是 P11
执行结果：P11 60 个正式 cells 仍为 0，完整 P0--P12 `12 + 2,134 = 2,146` suite 也未运行，
`paper_ready=false`。

当前原子性执行证据为
[`20260829T102845Z_p11_atomic_small_scheduler_smoke`](../results/20260829T102845Z_p11_atomic_small_scheduler_smoke/)：
`small / recompute / seed=2026082900` 选择 workflow
`4cb074e3cbf03d1709e2b44cbe575807813b`，完整 57 requests、57/57 completed，manifest 明示
`atomic_workflow_replay=true` 与 `smoke_request_truncation_permitted=false`。suite manifest SHA-256 为
`697f88ce4bd99626b8c0246d683bc59ed1f91d49ad75d55b1e87ad9eae5eb161`。该 cell 没有 Aegis protection/
recovery，故障未命中 active request，仍是 modeled/preliminary/non-paper-ready；它只证明完整 workflow
selection 和 Scheduler replay，不是 P11 paired performance 或 predictor coverage 结果。
