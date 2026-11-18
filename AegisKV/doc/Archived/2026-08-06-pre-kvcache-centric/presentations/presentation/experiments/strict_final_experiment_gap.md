# AegisKV Strict-Final 实验缺口

更新日期：2026-07-02

当前 `validate_figure_evidence.py --write-hpca-hybrid --strict-hpca-hybrid`
显示 HPCA hybrid gate 已经是 25/25 ready。这里的 hybrid-ready 允许使用
SGLSim online evidence 加上小规模 SGLang/torch CUDA 物理 copy 校准；它不等于
strict-final。strict-final 仍要求把关键 preliminary 图替换为端到端物理 KV
movement 或真实 recovery path 证据。

## 本轮新增证据

Fig. 3c `fig_online_alpha_adaptation` 已从单次 online trace + MDP replay 升级为
online adaptive-alpha sweep：

- 运行目录：
  `sglsim_large/results/agentic_workflows/sglsim-staged-13131-alpha-online-sweep-20260702`
- workflow：1-3-1-3-1。
- sweep point：low/high workload pressure，多个 failure-rate state。
- 关键观测：Critical agents 在 low pressure 下选择较低 GPUReplica alpha，在
  high pressure 下选择更高 alpha；Non-Critical agents 的 effective GPUReplica
  alpha 保持为 0。
- 证据层级：`hybrid-online-alpha-sweep`。它已经是在线 serving sweep，但还不是
  physical KV movement strict-final。

新增 current physical-recovery smoke，用于避免继续依赖旧的手工环境变量组合：

- runner：
  `sglsim_large/scripts/run_current_physical_recovery_smoke.py`
- 运行目录：
  `sglsim_large/results/agentic_workflows/sglsim-current-physical-recovery-smoke-20260702`
- 环境：`SGLSIM_AEGIS_RECOVERY_MODEL=physical_copy`，并读取
  `sglang_small/results/physical_recovery_path_aggregate/physical_recovery_path_aggregate.csv`。
- 关键观测：`8/8` requests completed，`4` 条 replay restore rows，`106`
  restored tokens，平均 restore sleep 为 `4.466 ms`。
- live movement audit：`4/4` restore rows 仍为
  `metadata_materialization`，timing source 为 `physical_recovery_path_csv`，
  `physical_tensor_copy` 行数为 `0`。
- 证据层级：`sglsim-current-physical-recovery-smoke-nonfinal`。它只是当前代码
  可重复的恢复时延和 request-detail instrumentation 门禁，不改变 P0
  strict-final 缺口。

同时，Fig. 10/Fig. 14 的 recovery-action-calibrated agentic suite 已重跑为当前
request-detail schema：

- 运行目录：
  `sglsim_large/results/agentic_workflows/sglsim-current-physical-recovery-action-agentic-r3-20260702`
- 场景：DP2 Host、DP4 peer、DP4 non-peer、DP4 peer hot-tail，每个场景 3 次。
- 关键观测：`1440/1440` requests completed，`288` 条 replay restore rows，
  Fig. 10/Fig. 14 measured rows 已切到该 run id。
- live movement audit：`288/288` restore rows 现在都有当前字段，全部为
  `metadata_materialization`，timing source 全部为 `physical_recovery_path_csv`，
  `physical_tensor_copy` 行数仍为 `0`。
- 影响：P0 readiness audit 不再包含旧 suite 的 `legacy_missing_nonfinal`，
  但 strict-final-ready 仍为 false。

同时，Fig. 8/Fig. 9 的 policy/recovery sweep 已切到 1-3-1-3-1 staged
workflow：

- 运行目录：
  `sglsim_large/results/agentic_workflows/sglsim-staged-13131-policy-stress-v2-r3-20260702`
- 场景：no backup、uniform Host、uniform GPU、AegisKV，每个 policy 3 个
  matched seeds。
- workload：DP4，27 个 agent 请求，`max_concurrency=9`，1-3-1-3-1 stage
  barrier，soft failure replay，Critical-only failover/retry，输出长度 128，
  failure after 128 output tokens。
- 关键观测：Fig. 8 直接从同一批 `request_details.csv` 计算 workflow-level
  total WCT 和 failure-exposed restore cost，而不是 per-request mean latency。
  normalized total WCT 为 no backup `1.000`，uniform Host `0.982`，uniform
  GPU `0.987`，AegisKV `0.993`；WCT 差异较小，因为 workflow 正常 serving
  时间仍占主导。failure-exposed restore cost 为 no backup `1.000`，uniform
  Host `0.760`，uniform GPU `0.742`，AegisKV `0.699`。Fig. 9 retry
  latency ratio 同样为 no backup `1.000`，uniform Host `0.760`，uniform GPU
  `0.742`，AegisKV `0.699`。AegisKV 在 135 条 replay action 中选择 72 条
  Host restore 和 63 条 peer-GPU restore。
- live movement audit：540 条 replay rows，405 条 restore rows，全部为
  `metadata_materialization`，timing source 为 `physical_recovery_path_csv`，
  `physical_tensor_copy` 行数仍为 `0`。
- 影响：Fig. 8/Fig. 9 已不再依赖旧的 generic policy sweep 或 replay
  counterfactual，但仍不是 strict-final physical KV movement。

## 剩余 strict-final 缺口

P0 online suites 的当前审计结果维护在
`sglsim_large/results/strict_final_readiness/p0_strict_final_readiness.md`。
截至 2026-07-02，当前 P0 suites 共 693 条 restore rows，其中 0 条是
`physical_tensor_copy`，因此 strict-final-ready 为 false。最新 audit 还验证了
这些 restore rows 的 request-detail 字段完整性：`missing_data_plane_rows=0`，
`missing_timing_source_rows=0`。因此 P0 blocked 的原因不是 schema 缺失，而是
restore data plane 仍为 `metadata_materialization`。

`refresh_strict_final_status.py` 现在会在生成 P0 calibration bridge 和
dataplane readiness report 之前先运行 `audit_p0_strict_final_readiness.py`。
因此每次 strict-final 状态刷新都会从当前 P0 `request_details.csv` 重新计算
`physical_tensor_copy_rows`、`nonfinal_restore_rows` 和 schema 完整性，避免使用过期
的 `p0_strict_final_readiness.csv`。

同时，refresh 也会运行 `write_p0_live_movement_audits.py`，为每个 P0 suite
生成 per-run 明细，并在
`sglsim_large/results/strict_final_readiness/p0_live_movement_audits.md` 中写入汇总。
当前汇总显示两个 P0 suite 合计 `restore_rows=693`、
`physical_tensor_copy_rows=0`、`nonfinal_restore_rows=693`，并且
`missing_data_plane_rows=0`、`missing_timing_source_rows=0`。
这进一步确认缺口是 data-plane 语义仍停留在 metadata materialization，而不是日志字段
或 audit artifact 缺失。
`report_strict_final_dataplane_readiness.py` 也会读取
`p0_live_movement_audits.json`，并检查它与 `p0_strict_final_readiness.csv` 的
restore/data-plane 计数是否一致；如果二者漂移，readiness 状态会变为
`p0_live_audit_mismatch`，不会继续给出 normal waiting 状态。

refresh 还会在生成 P0 bridge rows 之后运行
`validate_p0_physical_tensor_copy_candidates.py`。这个脚本只接受逐请求的
live `physical_tensor_copy` 候选证据：`suite_label`、`request_csv`、`rid` 必须能
对上当前 P0 bridge row，`aegis_replay_restore_data_plane` 必须是
`physical_tensor_copy`，`aegis_replay_restore_timing_source` 不能是 calibration CSV，
并且必须带有 `physical_copy_verified`、`copy_latency_ms`、`copy_bytes` 和
`source_artifact`。当前 candidate audit 状态会被
`strict_final_dataplane_readiness.md` 单独展示；它不会把现有
`metadata_materialization` rows 自动提升为 strict-final。

为避免未来手工拼 CSV 出错，新增
`build_p0_physical_tensor_copy_candidates.py` 作为 live-copy event 到 candidate CSV
的转换入口。该脚本读取 `p0_physical_bridge_rows.csv` 和一个或多个
`--event-csv` / `--event-jsonl`，只为通过校验的 live copy event 输出
`p0_physical_tensor_copy_candidates.csv`：event 必须带有当前 P0 bridge row 的
`suite_label/request_csv/rid` 主键，必须显式设置
`aegis_replay_restore_data_plane=physical_tensor_copy`，或提供
`event_kind=live_physical_tensor_copy` / `live_host_h2d_copy` /
`live_peer_p2p_copy` 这类 live physical-copy 身份字段，并且必须提供正的 copy
latency 和 copy bytes。builder 输出的 generation audit 和 candidate rows 会标记
`event_schema_version=p0-live-copy-event-v1`，用于区分后续可能变化的实机事件格式。
随后仍由
`validate_p0_physical_tensor_copy_candidates.py` 再次检查完整性；builder 本身不会
绕过 strict-final gate。
如果 live-copy event 已经同步到本地，也可以直接通过
`refresh_strict_final_status.py --p0-candidate-event-csv <events.csv>` 或
`--p0-candidate-event-jsonl <events.jsonl>` 让 refresh 在 validator 之前自动运行
builder；未提供 event 输入时 refresh 不会生成空 candidate CSV，从而保留当前缺口的
真实状态。
如果显式传入的 event 路径不存在，builder audit 会记录
`status=missing_event_sources` 和 `missing_event_sources`，用于区分“尚未提供
live-copy event”和“实验同步路径配置错误”。
为减少后续接入真实事件时的字段漂移，refresh 还会生成
`sglsim_large/results/strict_final_readiness/p0_physical_tensor_copy_events.template.csv`
和
`sglsim_large/results/strict_final_readiness/p0_physical_tensor_copy_event_schema.md`。
这两个文件只描述 `p0-live-copy-event-v1` 的输入格式和表头，不包含任何候选证据行，
也不会让 candidate audit 变为 ready。
builder 和 validator 还会记录 `bridge_required_digest`，这个 digest 由当前
`p0_physical_bridge_rows.csv` 中每个 required restore row 的
`suite_label/request_csv/rid/restore_action/restored_tokens/physical_action` 计算得到。
`report_strict_final_dataplane_readiness.py` 会重新计算当前 bridge digest，并检查
candidate/generation audit 是否与当前 bridge rows 一致；这样即使 required row 数量
相同，也不会误用旧 workload 或旧 run 生成的 live-copy evidence。

运行时代码也显式保持这一边界：
`sglsim.srt.simulation.recovery_model.aegis_restore_data_plane()` 会把
Host/GPUReplica restore 标记为 `metadata_materialization`。即使启用
`SGLSIM_AEGIS_RECOVERY_MODEL=physical_copy`，该模式也只改变
`aegis_replay_restore_timing_source` 和 restore sleep，不会把数据面标记成
`physical_tensor_copy`。因此 strict-final 不能通过打开 physical-copy timing model
获得；必须接入真正的 live Host H2D / peer P2P KV movement，或同步等价的真实系统
recovery path 证据。

| 优先级 | 图 | 当前证据 | strict-final 还缺什么 |
|---|---|---|---|
| P0 | Fig. 8/9/10 | online SGLSim policy/recovery sweep + calibrated timing | 使用 physical tensor KV movement 的端到端 failover/retry policy sweep，并保留 per-request recovery path counters |
| P0 | Fig. 11b | Host-visible/invisible paired anchors + SGLSim rows | 相同写入成本下的真实 Host-visible recovery path 重复测量，证明 failure 时 frontier materialization 的收益 |
| P0 | Fig. 15 | SGLang/SGLSim calibration anchors | 同一 AegisKV failure-policy workload 上的真实 SGLang recovery 与 SGLSim 对齐，而不仅是 base serving timing |
| P1 | Fig. 3/12/13/14/14b | online SGLSim budget/ablation/DP/peer-route sweep | 用 physical KV movement data plane 重跑主要 policy matrix，替换 metadata-materialization preliminary rows |
| P1 | Fig. 2/15b/15c | SGLang/torch copy microbench + recovery model check | 接入 live retry KV buffer 的端到端 restore path，而不是只校准 copy component |
| P2 | Fig. 1/1b/1c/3d | 1-3-1-3-1 stage-barrier trace-derived motivation | 小规模真实 SGLang agentic serving calibration，确认 Critical/Non-Critical WCT 暴露趋势与 SGLSim 一致 |
| P2 | Fig. 11 | no-failure overhead anchors | 重复 matched SGLang/SGLSim normal-path overhead，覆盖 AegisKV policy on/off 和代表性并发 |

## 建议下一步

1. 先实现或启用 live physical KV movement 的 retry data plane，让 SGLSim policy
   sweep 不再只是 metadata materialization。
2. 用同一 1-3-1-3-1 workload 重跑 P0 policy/recovery sweep，优先更新 Fig. 8/9/10
   和 Fig. 11b。
3. 再扩展到 budget、ablation、DP scale、peer-rank selection，把 Fig. 3/12/13/14/14b
   从 hybrid-ready 升级为 strict-final-ready。
