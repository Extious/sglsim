# SGLang 本机 `torch.profiler` 与 D2H Backup 干扰实验

## 结论

本目录保存的是 `gpu10` 上的本机实测，不是历史 trace。

- 在单请求、TP=1、eager/no-CUDA-graph 的 Qwen3-0.6B 实验中，选中的
  prefill step 内 kernel-union 占比中位数为 **92.51%**；decode step
  只有 **8.15%**。这说明本配置的 prefill 很密，而 eager decode 明显
  launch-limited。
- 但是 decode 中的 trace 空隙不能直接当成安全 Backup 窗口。连续
  16 MiB pinned D2H chunk 达到 25.416 GB/s 时，64-token decode 延迟相对
  前后两段 control 的中位数增加 **8.08%**。
- 因而 AegisKV 不应采用“看到 GPU kernel 空隙就拷贝”的规则。更可靠的
  起点是：以 decode 为优先候选窗口，将 Backup 切成可抢占 micro-chunk，
  同时用前台 SLO/interference feedback 控制速率；prefill 中默认暂停或
  极低速执行。

## 运行身份与环境

- 原 checkout：`/home/comp/zhanzhao/sglang`
  - 分支：`multi-level-backup`
  - commit：`dce5a12be0a4a084f341beec7da9d28a2c3492dd`
  - 实验前后均未切换，审计时 worktree clean。
- 实际运行源码：从上述仓库建立的 detached 临时 worktree
  - 分支 ref：`profiler`
  - commit：`fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1`
  - 实验后已移除临时 worktree。
- 主机：`gpu10`
- GPU：物理 GPU 1，NVIDIA GeForce RTX 4090 24 GiB
- Driver：`610.43.02`
- Python：3.12.12
- PyTorch：2.9.1+cu128
- 本地完整模型：
  `/home/comp/zhanzhao/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca`
- Server：TP=1，context=8192，`mem-fraction-static=0.35`，
  Triton attention，PyTorch sampling，prefill/decode CUDA graph 均关闭。

硬件并非独占，实验期间其他用户在两张卡上有进程。因此本结果是机制定位
证据，不是可发表的最终性能数字。

## 兼容性处理与已知失败

`profiler@fdebc938f7` 比本地环境更新，声明需要 torch 2.11 和
sglang-kernel 0.4.5，而本地为 torch 2.9.1 和 sglang-kernel 0.4.1。
为了不修改用户 checkout 或 `.venv`：

1. 使用独立 detached worktree，并以 `PYTHONPATH` 指向该 worktree。
2. 此 branch 的 `Cohere2MoeConfig` 会被 huggingface_hub 1.10.2 的
   `@strict` 拒绝。`sitecustomize.py` 仅在进程内让它走 branch 已有的
   “older huggingface_hub” fallback；未 patch repo。该问题后来由 upstream
   commit `14c0a31829` 修复。
3. 设置 branch 提供的 `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`。因此本次
   只能证明该小模型/后端在此组合下真实执行，不能证明 profiler branch
   所有 kernel 与声明依赖完全兼容。
4. 首次 warmup 因子进程 PATH 中找不到可执行 `ninja` 而失败；将现有
   `.venv/bin` 加入 PATH 后，server 成功加载权重、分配 KV cache、完成
   warmup 并对真实请求返回结果。
5. capture helper 在 scheduler 达到 `num_steps` 后自动停止 profiler，
   随后又调用 `/stop_profile`，得到一次
   `Profiling is not in progress` 500。两份 trace 已在自动停止阶段完整
   flush，analyzer 随后正常解析并退出 0；`live_capture.stderr.log` 为空。

## 可复现命令

核心命令如下；临时 worktree 名称可变化。

```bash
git -C /home/comp/zhanzhao/sglang worktree add --detach \
  /tmp/sglang-profiler-worktree profiler

export PATH=/home/comp/zhanzhao/sglang/.venv/bin:$PATH
export CUDA_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export PYTHONPATH=/home/comp/zhanzhao/sglsim/AegisKV/flowchart/ideas/evidence/sglang_profiler:/tmp/sglang-profiler-worktree/python

/home/comp/zhanzhao/sglang/.venv/bin/python -m sglang.launch_server \
  --model-path /home/comp/zhanzhao/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --host 127.0.0.1 --port 30177 --tp-size 1 \
  --mem-fraction-static 0.35 --context-length 8192 \
  --max-running-requests 16 --disable-cuda-graph \
  --attention-backend triton --sampling-backend pytorch

python analyze_llm_torch_profile.py \
  --framework sglang --url http://127.0.0.1:30177 \
  --output-dir .../live_stage_profiles \
  --profile-prefix aegiskv-local-4090 \
  --num-steps 5 --warmup-steps 3 --profile-by-stage \
  --profile-workload both \
  --prefill-input-len 2048 --prefill-output-len 1 \
  --decode-input-len 1 --decode-output-len 256

python analyze_stage_timeline.py \
  --prefill live_stage_profiles/prefill/...trace.json.gz \
  --decode live_stage_profiles/decode/...trace.json.gz \
  --output stage_timeline_metrics.json
```

D2H intervention：

```bash
CUDA_VISIBLE_DEVICES=1 python d2h_interferer.py \
  --duration-s 40 --buffer-mib 256 --chunk-mib 16 \
  --output d2h_interference_run.json
```

`run_inference_latency_probe.py` 使用随机、互不相同的 token ID；所有请求
`cached_tokens=0`，避免 radix prefix cache 污染对照。

## Trace 与 kernel 结果

### 原始 trace

| Stage | 文件 | 压缩大小 | SHA256 |
| --- | --- | ---: | --- |
| Prefill | `live_stage_profiles/prefill/1785256905.704325/aegiskv-local-4090-prefill-1785256905.9262657-TP-0.trace.json.gz` | 32,358,699 B | `5bb98263f1e0474efcf0c1ce697f64cd2c491d690e5710329e5ea9bb3bb8286b` |
| Decode | `live_stage_profiles/decode/1785256939.221301/aegiskv-local-4090-decode-1785256939.273685-TP-0.trace.json.gz` | 32,628,258 B | `dfb3aaa14faf2e2325b51b51810f8543d45debb120542628e69c97729fbc7ed5` |

未压缩大小分别约 382.3 MB 与 386.0 MB。branch 自带 analyzer 的完整三表在
`live_analysis.txt`：

- Prefill：BF16 GEMM 占 47.6%，extend-attention kernel 占 31.5%。
- Decode：GEMV 占 78.3%，fused add+RMSNorm 占 4.3%，attention 的
  grouped/reduce kernels 合计约 6.2%。

### Step-filtered timeline

capture 的 `num_steps=5` 是所有 forward step 计数。为了不把短 decode 混进
prefill，`stage_timeline_metrics.json` 只选择 `step[EXTEND ...]` 或
`step[DECODE ...]`：

| 指标 | Prefill | Decode |
| --- | ---: | ---: |
| 选中 step 数 | 3 | 5 |
| step duration 中位数 | 24.013 ms | 22.819 ms |
| kernel-union busy 中位数 | 22.219 ms | 1.850 ms |
| kernel-union occupancy 中位数 | 92.51% | 8.15% |
| 无 kernel/memcpy/memset gap 中位数 | 0.672 µs | 42.593 µs |
| gap p95 | 10.848 µs | 161.985 µs |
| 最大 gap | 514.852 µs | 2,304.111 µs |
| CPU kernel-launch issue gap 中位数 | 37.773 µs | 42.432 µs |
| CPU kernel-launch issue gap p95 | 252.109 µs | 285.256 µs |

Prefill 的 1,152 个 observed gaps 中，59 个至少 10 µs、15 个至少
100 µs、1 个至少 500 µs。Decode 的 1,735 个 gaps 中，1,558 个至少
10 µs、183 个至少 100 µs、4 个至少 500 µs、3 个至少 1 ms。

选中的 prefill step 内只有 3 个 D2D copy、总计 24 B；decode step 内没有
memcpy。TP=1 trace 中没有 NCCL hardware event。准确含义只是“该 SGLang
进程在这些 step 中没有发出 D2H/NCCL op”，不能据此宣称 copy engine、
PCIe、HBM、NIC 或全机 fabric 空闲。

## Pinned D2H 干扰 A/B

Interferer 使用一个 256 MiB device buffer 和一个 256 MiB pinned host
buffer，以独立 CUDA stream 连续执行 16 MiB D2H，每个 chunk 同步一次：

- 持续：40.000 s
- 拷贝：60,597 chunks，1,016,648,957,952 B
- 实测吞吐：25.416 GB/s

请求延迟为客户端 wall-clock；所有请求均 `cached_tokens=0`。

| Workload | Baseline | D2H interference | Post-baseline | 干扰相对前后 control 中位数均值 |
| --- | ---: | ---: | ---: | ---: |
| Prefill 2048→1，n=10/组，中位数 | 28.691 ms | 24.803 ms | 25.434 ms | -8.35% |
| Prefill p95 | 41.613 ms | 37.335 ms | 42.280 ms | — |
| Decode 1→64，n=6/组，中位数 | 677.124 ms | 718.677 ms | 652.764 ms | **+8.08%** |
| Decode p95 | 700.105 ms | 747.553 ms | 673.566 ms | — |

Prefill 没有观察到 slowdown，不能解释成“D2H 免费”：请求很短，存在 GPU
升频、顺序和共享机器噪声。Decode 的干预方向在 mean、median、p95 上一致，
说明持续饱和 D2H 会通过 copy engine/HBM/PCIe 或调度竞争影响前台；但
n=6 且 GPU 非独占，幅度仍需在独占卡上随机交错复测。

## 对 Backup 时机的可用结论

1. **Prefill 默认不重叠。** 本配置 prefill 的 kernel union 已占约 92.5%，
   p95 observed gap 只有 10.8 µs。除非有独立 counter 证明通道余量，否则
   Backup 应暂停或仅执行极小 control chunk。
2. **Decode 是候选而不是保证。** eager decode 暴露较多 launch gaps，但
   连续 D2H 仍使 decode 中位延迟上升约 8%。需要 deadline-aware
   micro-chunk、可抢占 copy stream 和 SLO feedback，而不是固定带宽。
3. 以本次 25.416 GB/s 的大 chunk 吞吐作乐观上界，decode median/p95 gap
   分别只容纳约 1.08/4.12 MB；prefill p95 约 0.28 MB。小 chunk 吞吐更低且
   有 launch overhead，所以实现起点宜更保守，例如 prefill 不超过 256 KiB、
   decode 不超过 1 MiB，并由在线干扰信号升降速。
4. `torch.profiler` 能证明 kernel、memcpy、NCCL event 的时间与进程内缺席，
   不能证明 HBM/PCIe/NVLink/RDMA 的剩余带宽，也不能观察其他进程全部活动。
   下一阶段需要 CUPTI/Nsight counters 和受控 copy intervention。
5. 该 trace 是 TP=1、单请求、graph-off。不能外推到 continuous batching、
   CUDA graph、TP/PP/DP/EP；尤其 TP/EP 的 NCCL/NVLink 竞争必须单独 profile。

## 清理状态

- SGLang server 已以 Ctrl-C 正常关闭；`127.0.0.1:30177` 已拒绝连接。
- D2H interferer 已自然结束。
- `ps` 未发现本实验的 server、interferer 或 analyzer 进程。
- detached profiler worktree 已确认 clean 后移除。
- 原 `/home/comp/zhanzhao/sglang` checkout 未切分支、未写入。
