# Qwen3-8B PP=2 torch.profiler 实验

本目录用于在一个节点的两张 RTX 4090 上，以 SGLang `profiler` 分支部署
Qwen3-8B，并分别捕获：

1. `chunked_prefill_size=1024` 下的长提示 prefill；
2. 稳态连续批处理 decode；
3. 两个 PP rank 的 CPU/GPU 函数与 kernel trace；
4. CUDA allocator 内存快照及浏览器 HTML；
5. 200 ms 粒度的 GPU 利用率、显存、功耗、温度、时钟、PCIe 链路和
   SGLang 进程树 CPU/RSS。

## 集群拓扑边界

2026-07-29 的 `scontrol` 结果显示，`short` 分区节点只有 L40S 与 A100；
RTX 4090 节点属于 `long`，其中 `gpu10-gpu13` 也属于 `debug`。因此作业允许
调度器在 `debug,long` 中选择，并严格请求 `--gres=gpu:rtx4090:2`；这保留
用户要求的 GPU 型号、单节点与 PP=2，只改变不存在的分区组合。作业产物中的
`environment.txt` 会再次保存实际节点、GPU UUID、型号、driver 和 Slurm
分区，避免把它误写成 short。

## 提交

```bash
sbatch run_qwen3_8b_pp2_profile.sbatch
```

每次运行写入 `runs/$SLURM_JOB_ID/`。`LATEST_RUN` 指向最近一次成功走到作业
末尾的运行。`index.html` 是可直接打开的资源利用 dashboard；
`browser_trace/pp2-prefill-decode-resources.trace.json.gz` 可一次导入
Perfetto (`https://ui.perfetto.dev`) 或 Chrome tracing，同时查看 PP0/PP1
的 chunked prefill、decode 和 GPU/CPU/PCIe counter。各 rank 原始
`*.trace.json.gz` 仍保留为权威 trace；`*-memory-timeline.html` 和
`*-allocator-history.html` 可直接由浏览器打开。

## 负载选择

负载分成三个层次，避免把“一次性突发”或 profiler 扰动误写成正常路径：

1. 先用并发 2/4/6/8/12/16/24/32/40/48 探测容量；
2. 再以探测峰值的 75%/85%/90%/95%/100%/105% 运行限速到达流；
3. 选择吞吐至少达到限速候选峰值 90%、校准期 scheduler queue 均值至少
   8 ms，
   且 queue p95 上界不超过 `max(50 ms, 25% × e2e p50)` 的第一个点；
4. 用选中速率另跑 20 秒 `steady_load`，它才是正常路径资源利用率和
   “近饱和、少量排队”的正式证据；该正式重复仍须满足 queue 均值至少
   2 ms。校准使用更高的 8 ms 下限，为节点抖动留下裕量。

排队时间来自每个 phase 前后
`sglang:queue_time_seconds` Prometheus histogram 的差值：均值由
`sum/count` 精确计算，p50/p95 是直方图 bucket 的保守上界。PP=2 HTTP
响应中的 request-time 字段不作为证据来源。若没有候选满足条件，会保留
fallback 诊断但最终 evidence gate 失败，不能进入 `LATEST_RUN`。

正式 Kineto 只发送一次带 `start_step` 和 100 个 forward 步数的 arm 请求，
由两个 PP scheduler 的本地 forward counter 自动停止：
prefill 波次使用 4096-token 输入，因此每个请求至少跨四个 1024-token
chunk；decode 波次使用 16 路并发、128-token 输出，确保观测稳定 decode
batch，而不是单请求空载 decode。第一波最多包含
`16 × (4096/1024) = 64` 个 chunk prefill step 和 16 个 decode step；第二波
短提示最多再占 16 个 prefill step。因此 100 步窗口必然跨过第一波，并至少
记录后一波 4 个稳定 decode step。每个 PP rank 生成一份同时覆盖两波的 combined
权威 trace；最终 Perfetto 文件只合并这两份 rank trace，并为每个来源分配
独立 PID/flow-ID namespace。
服务器使用
`mem_fraction_static=0.80`，为最坏并发 32 的 PP 单阶段 KV 留出空间；任一
正式请求发生 scheduler retraction 都会让最终 evidence gate 失败。
profile phase 的排队仍会报告，但由于 torch.profiler 的 CPU/GPU/shape
采集会放大运行时开销，它只用于解释 trace 扰动，不替代 `steady_load`。

CUDA allocator 先以独立的 MEM-only 小波次采集，再执行唯一一次 Kineto arm。
这是针对正式作业 `63329`/`63437` 复现出的 PyTorch 原生析构问题：连续
创建带 stack unwinding 的 Kineto 结果，或在其后触发 allocator snapshot，
会在延迟 GC 中段错误。正式 trace 使用 `with_stack=false`，关闭的是每个 op
附带的 Python 源码调用栈；CPU function/record-function、ATen、CUDA runtime、
GPU kernel、NCCL、shape 以及整机 counter 时间线仍完整保留并进入分析。
作业 `63466` 进一步证明 PP=2 下不能用手动 `/stop_profile` 导出：PP0 进入
同步 stop/export 路径后长时间不返回，而停止控制尚未使 PP1 进入导出；
清理阶段只留下缺少 gzip 结束标记的 PP0 截断文件，PP1 文件缺失。
按 forward 数自动停止由两个 scheduler 在各自的 pipeline
forward counter 上触发，不依赖第二条 stop 控制消息，因此规避该路径死锁；
正式重跑只保留这一 arm，不再启动后续 Kineto。

服务端口由 `SLURM_JOB_ID` 派生，避免共享节点上的固定端口冲突；trace 与
遥测采集完成后会先关闭 SGLang 模型进程、释放 HBM，再执行 CPU 侧 trace
合并、allocator HTML、triage 和报告生成。

作业当前排除 `gpu14`：2026-07-29 13:57 后该节点两张卡被 Slurm 作业外的
`ollama/llama-server` 占用。每次启动还会检查分配卡必须为空，并把结果写入
`gpu_processes_before_launch.txt`；若发现外部占用则以退出码 20 失败，而不
与未知进程争抢显存。运行期间资源采样器还会每秒把 GPU compute PID 与
SGLang 进程树对账；正式 phase 内出现任何 foreign GPU PID，同样不能进入
`LATEST_RUN`。

`LATEST_RUN` 只在 `analysis_summary.json` 的 evidence gate 和
`FINAL_VALIDATION.json` 的 artifact contract 都通过后写入。后者会核对精确
部署参数、两份 PP-rank 原始 combined trace、单文件合并 trace、两份 stage
JSON、两份 combined triage、allocator HTML、dashboard 内容和正式 workload
shape。工作负载侧还会完整读取 gzip 流确认 end-of-stream marker，避免把
`63466` 一类大小非零但导出中断的 trace 当作就绪文件。

正式修复版作业 `67111` 已于 2026-08-03 在 `gpu10` 完成并进入
`LATEST_RUN`：Slurm exit `0:0`，正式稳态 109/109 请求成功，PP0/PP1 各生成
一份完整 combined trace，浏览器 trace、allocator HTML、stage/triage、资源
dashboard、checksums 与最终合同全部通过。提交脚本还会在 SGLang 后台进程于
`/health` 就绪前退出或超过 900 秒健康期限时记录诊断并立即失败，避免作业在
服务已经消失后继续空等。
