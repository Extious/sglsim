# 09. 结论与下一步

## 结论

AegisKV should be framed as a workflow-aware, restore/recompute-aware KV backup layer for multi-agent LLM serving. Its novelty is not that it stores KV in another tier. In this simplified model, SGLang already performs fixed `write_through` D2H after radix insertion; AegisKV makes the failed rank's surviving Host KV visible to retry ranks, decides which workflow-critical radix segments deserve an additional GPUReplica, chooses the peer rank, and decides whether failure recovery should restore or recompute each suffix.

The paper's core sentence should be:

> In multi-agent LLM serving, the KV state worth protecting is not the largest, newest, or globally most reused state; it is the state whose loss most increases workflow completion time after accounting for restore-vs-recompute cost and backup interference.

## 近期下一步

1. Finalize the Host definition as same DP rank / worker local Host memory, not same-GPU duplicate.
2. Implement or verify SGLang metadata path:
   `request -> Req -> radix node -> Host/peer ACK -> failure retry`.
3. Implement fixed `write_through` HostBackup directory: ACKed failed-rank Host KV must be discoverable by retry ranks.
4. Implement peer-rank selection with load + KV locality + retry route + HBM/P2P pressure.
5. Add shared-prefix SafeCover accounting and frontier invalidation on Host/peer eviction.
6. Use the restored `benchmark/agentic_workflows` wrapper to run repeated online workflow scenarios, then replace its preliminary rows with final AegisKV policy evidence.
7. Generate the motivation equal-size-loss figure.
8. Generate restore-vs-recompute microbenchmarks for Host and GPUReplica.
9. Run the four-policy online suite:
   `none,host,peer_gpu,aegiskv`.
10. Compare SGLang and SGLSim on the same small workloads.
11. Only after those pass, run DP=8 / 30K-token SGLSim scale-up.

## 最终 HPCA 论文前必须补齐的证据

- Aegis metadata does not affect normal SGLang outputs.
- Host ACK creates a safe frontier and retry can restore from it.
- GPUReplica peer copy creates a safe frontier and retry can use it.
- GPUReplica peer selection beats random, least-loaded-only, and locality-only placement.
- Shared-prefix marginal accounting reduces duplicate protected bytes without reducing recovery coverage.
- Host-visible AegisKV beats Host-invisible fixed-write-through baseline under the same D2H cost.
- Radix lock/split/eviction tests reject unsafe frontiers.
- AegisKV beats static baselines on WCT under failures.
- AegisKV keeps no-failure overhead bounded.
- Restore-vs-recompute decision improves over restore-all and recompute-all.
- SGLSim preserves policy ranking relative to SGLang.

## 暂时不能声称的结论

Do not claim:

- AegisKV handles host failure. Host tier does not survive full host loss.
- AegisKV is a generic RemoteStore or durable KV system.
- AegisKV replaces SGLang's KV manager.
- AegisKV requires DRL. The first paper should be explainable and classical.
- Offline counterfactual results are equivalent to online runtime results.

## 未来扩展

After the two-tier paper is stable:

- add RemoteStore for host failure；
- add DRLPart-style budget controller；
- add failure-prediction-aware exposure；
- add cross-node restore scheduling；
- add application-level checkpoint / summary interactions；
- evaluate with real Codex/CrewAI/LangGraph traces。
