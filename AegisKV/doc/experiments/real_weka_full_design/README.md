# Full real-Weka AegisKV simulation

> **Invalidated calibration:** the archived `four_path_v2` replay is no longer
> admissible as current AegisKV performance evidence. Its supposed Peer-HBM
> calibration used Mooncake TCP while reading and writing registered GPU
> memory, rather than CUDA P2P/D2D. The replay remains diagnostic-only and
> must be regenerated from a fresh current-source V100S physical-calibration
> lineage; the terminal historical v11 chain described below is ineligible.
> Independently, 85,129 of 98,827 unscaled requests exceed Qwen3-8B's
> 40,960-token model limit, so the archived replay is not model-executable.
> See `../transport-invalidation-rtx4090-20260823.json`.

### Independent RTX4090 PP2 lineage (targeted diagnostic only)

A newer physical campaign is imported through a separate contract at
[`RTX4090_PP2_LINEAGE.md`](RTX4090_PP2_LINEAGE.md).  It is identified only as
`RTX4090-2node-PP2-systemDP2` (Qwen3-8B, TP=1, PP=2, system-DP=2) and is never
accepted by the V100-only importer or `run.py` preflight.  The campaign's 64
stage-local source-copy completions were driven by HBM agents whose start
events all recorded Mooncake `tcp`; a bounded two-node topology diagnostic (job 135738,
COMPLETED/0:0 in four seconds) observed P2P `NS` and bidirectional
`torch.cuda.can_device_access_peer=false` on both nodes.  Consequently the
lineage is physical registered-HBM protection/recovery over Mooncake TCP—not
CUDA P2P/D2D or GDR—and its rate is not default simulator calibration.

The independent consumer defaults to validation only and records
`default_replay_authorized=false`, `formal_ready=false`, and
`paper_ready=false`.  Even its explicit TCP acknowledgement only constructs a
metadata timing object; it has no replay or figure entry point.  No current
V100 schema, compatibility identity, runtime contract, or stable mechanism
check name is changed or relaxed by this lineage.

#### Current immutable RTX4090 PP2 reporting snapshot (record only)

The current reporting snapshot for this independent lineage is
`/home/comp/zhanzhao/Mooncake/aegiskv/analysis/rtx4090-pp2-long-context-preliminary-v2/paired-series-audit.json`,
SHA-256
`f5223bcbb3f35ee497ea4ddd08932fb0a31d7014ffd136f7e16c6c260c5606e0`.
Its figure audit is
`figure/paired-series-figure-audit.json`, SHA-256
`f01fd8e34cb80364bccd94ee34a7b9ff680e412770f0b308c94487bef51b448d`.
Both files and all four referenced pair reports were re-hashed before this
record was updated.

The snapshot retains jobs 135535, 135567, 135599, and 135601 with seeds
20260822--20260825, respectively.  It has two `baseline-first` and two
`aegis-first` pairs, retains all four valid pairs, and reports 4/4 positive
failure-added-latency differences.  The mean saving is 139.814 ms (6.765%),
the paired-t 95% interval is 52.446--227.182 ms, and the mean healthy-path
overhead is 23.876 ms.  These statistics describe only the predeclared
four-pair long-context targeted diagnostic.

The current raw-path diagnostic is
`/home/comp/zhanzhao/Mooncake/aegiskv/analysis/rtx4090-pp2-protection-pipeline-diagnosis-v1/protection-pipeline-audit.json`,
SHA-256
`52f9cfcab0fb98b695954ecddcc2212145f280620b1ac68b4d5331fa6172328c`;
it counts 64 stage-local source-copy completions. Across eight Aegis cells, ordinal-zero
source completion averages 99.493 ms versus 35.196 ms later, while the two
target stages average 69.101 ms for their first inbound versus 32.340 ms
later. This source/target agreement establishes a warm-up-shape diagnostic,
not an A100/V100 coefficient. The same diagnostic records zero direct DRAM PUT
and zero AIMD pressure/reclaim, so it cannot calibrate those mechanisms.

All audited HBM agent starts used Mooncake TCP, and neither tested RTX4090 node
had bidirectional CUDA peer access.  The snapshot therefore keeps
`formal_ready=false`, `paper_ready=false`, and
`simulator_rate_default_eligible=false`: it is not CUDA P2P/D2D or GDR
evidence, is not a default SGLSim physical calibration, and is not an input to
the V100-only consumer.  The older v1 jobs 127169, 127205, 127265, and 127207
remain a separate diagnostic lineage and must not be pooled with or appended
to this v2 series.  This documentation update imported no artifact, ran no
replay, regenerated no figure, and submitted no performance work.

This experiment replays the complete `cc-traces-weka-062126` materialized
corpus through the current AegisKV metadata implementation. The full profile is
393 workflows, 98,827 LLM requests, 21,635,381,376 input tokens, 106,474,498
output tokens, and 5,697,618 progressively sealed KV chunks.

The replay covers all three current algorithms:

1. causal, workflow-disjoint dual-head GBDT remaining-time prediction followed
   by online fan-out/join CPM Slack; missing predictions fail closed to zero;
2. independent source-owned Peer-HBM D2D and Store-DRAM D2H SlackLandlord
   paths, per-tier PENDING/COMMITTED/ABORTED state, continuous prefixes, true KV
   bytes, lazy Slack reprioritization, target rotation, and terminal cleanup;
3. per-target HBMQuotaAIMD control windows.

### Current-design simulator execution logic (static evidence)

The current metadata execution path is mapped to the six
`source_owned_independent_hbm_dram_v2` requirements below. This is a static
code-and-unit-test result; it is not a replay or physical-performance result.

| Current-design requirement | Exact SGLSim implementation | Focused tests |
| --- | --- | --- |
| Visible residual DAG and causal Slack; incomplete or non-causal evidence fails closed to Slack 0, and Slack changes update metadata only | `python/sglsim/srt/simulation/aegis_residual_dag.py` (`build_residual_dag_slack_tickets`, `resolve_slack_for_design`), `python/sglsim/srt/simulation/aegis_full_design.py` (`_resolve_residual_slack`, `update_slack_from_ticket`, `prioritize_recovery`), and `python/sglsim/srt/managers/data_parallel_controller.py` (`_effective_aegis_slack_ms`) | `test/unit/srt/test_aegis_residual_dag.py`; `test_current_design_controller_fails_unproven_slack_closed_to_zero` and `test_current_design_controller_consumes_visible_residual_dag_ticket` in `test/unit/srt/test_data_parallel_controller.py`; `test_slack_decays_at_chunk_seal_without_triggering_copy_movement`, `test_full_runtime_fails_unproven_positive_slack_closed_to_zero`, `test_recovery_priority_uses_decayed_visible_residual_dag_slack`, and `test_slack_ticket_update_never_queues_or_moves_kv` in `test/unit/srt/test_aegis_full_design.py` |
| Independent HBM/DRAM admission with HBM_ONLY, DRAM_ONLY, HBM_AND_DRAM, or neither; no cross-tier future cost and no HBM prerequisite | `python/sglsim/srt/simulation/aegis_independent_tiers.py` (`_try_admit`, `admit_chunk`, `Residency`) | `test_independent_admission_exercises_all_four_residency_outcomes`, `test_each_tier_gain_is_unchanged_by_the_other_tier_future_path`, `test_hbm_only_is_valid_without_a_dram_prerequisite`, and `test_tier_ablation_does_not_create_cross_tier_dependency` in `test/unit/srt/test_aegis_independent_tiers.py` |
| Source-owned Peer-HBM and direct source-to-DRAM paths with independent, tier/path-bound ACK and recovery; no target-owned or HBM-to-DRAM path | `python/sglsim/srt/simulation/aegis_independent_tiers.py` (`CopyCompletionProof`, `complete_copy`, `completion_proof`, `recover_after_source_failure`) and `python/sglsim/srt/simulation/aegis_transfer_engine.py` (`TransferSubmission`) | `test_one_copy_failure_does_not_abort_the_other_tier`, `test_completion_ack_is_bound_to_exact_tier_path_and_pending_record`, `test_hbm_ack_fails_closed_on_each_path_identity_tamper`, `test_recovery_ignores_pending_and_uses_fastest_committed_prefix`, `test_recovery_prefers_peer_hbm_when_both_tiers_are_committed`, and `test_completion_proof_is_route_layout_and_all_shard_strict` in `test/unit/srt/test_aegis_independent_tiers.py`; `test_transfer_engine_rejects_policy_or_legacy_path_semantics` in `test/unit/srt/test_aegis_transfer_engine.py` |
| Per-pool two-level victim indices expose one safe, contiguous request-tail frontier | `python/sglsim/srt/simulation/aegis_independent_tiers.py` (`VictimFrontier`, `_push_tail`, `_pop_selection_tail`, `victim_frontiers`) | `test_each_pool_exposes_only_one_safe_tail_per_request` and `test_slack_landlord_replaces_lower_value_pool_local_tail` in `test/unit/srt/test_aegis_independent_tiers.py` |
| AIMD changes only target HBM quota, pauses HBM admission, and directly reclaims low-value safe HBM tails, including HBM_ONLY without DRAM backing | `python/sglsim/srt/simulation/aegis_independent_tiers.py` (`_force_hbm_victims`, `control_hbm_quota`) and `python/sglsim/srt/simulation/aegis_full_design.py` (`control_hbm_quota`) | `test_hbm_quota_aimd_can_directly_reclaim_hbm_only_tail` in `test/unit/srt/test_aegis_independent_tiers.py` |
| Storage control selects victims; the transfer engine only queues and independently rate-limits the two physical lanes | `python/sglsim/srt/simulation/aegis_independent_tiers.py` (`_select_victims`) and `python/sglsim/srt/simulation/aegis_transfer_engine.py` (`AegisTransferEngine.enqueue`, `control_boundary=queue_and_rate_limit_only`), wired by `AegisFullDesignRuntime._schedule_transfer` in `python/sglsim/srt/simulation/aegis_full_design.py` | `test_transfer_engine_has_independent_hbm_and_dram_rate_lanes`, `test_transfer_engine_rejects_policy_or_legacy_path_semantics`, and `test_transfer_submission_schema_has_no_victim_selection_field` in `test/unit/srt/test_aegis_transfer_engine.py`; `test_independent_transfer_completion_and_request_reclaim_wait` in `test/unit/srt/test_aegis_full_design.py` |

The focused static suite comprised
`test_aegis_residual_dag.py`, `test_aegis_transfer_engine.py`,
`test_aegis_independent_tiers.py`, `test_aegis_full_design.py`,
`test_aegis_independent_tier_manager.py`, `test_data_parallel_controller.py`,
`test_aegis_bounded_slack.py`, and `test_aegis_recovery_laxity.py` under
`test/unit/srt/`; it passed **148/148**. The V100-only consumer, schema, and ten
stable mechanism check names were not changed. SGLSim remains a metadata
simulator: this evidence performs no CUDA/NCCL operation and moves no physical
KV bytes. An earlier broad pytest glob accidentally invoked two synthetic
`run(config)` calls under pytest temporary directories; neither left a retained
result, output, figure, or performance artifact.

The checked-in template deliberately retains a diagnostic active-primary-
request proxy. The importer replaces it only when given a PASS, same-campaign
V100S pressure-calibration artifact. That artifact fits fixed non-shadow CUDA
residency from pre-control samples and preserves the runtime's exact predicate:
free CUDA headroom below its threshold OR a target-local HBM allocation/reclaim
operation at or above its preregistered stall threshold. The metadata replay
cannot reproduce allocator wall-clock durations, so it consumes the physical
stall probability conditioned on a window containing at least one local HBM
operation; deterministic per-target/window draws are shared across policies.
Peer-HBM transfer duration is never a pressure input. Legacy post-reclaim-only
samples, missing local thresholds, missing stalled/non-stalled operation
classes, dropped samples, or a changed artifact fail closed.

Eight cluster fault-window snapshots plus 512 request-stratified fail-now
snapshots compare six paired policies, separating placement and remaining-Slack
due-time ordering. A separate mutating probe for each source rank executes
fence/freeze, pending-copy abort, target activation, route-version advance,
HBM/DRAM/recompute selection, silent replay, all-PP/TP-rank completion, and
`CompleteRequest` cleanup.

The archived four-path run and its diagnostic interpretation are summarized in
[`RESULTS.md`](RESULTS.md). Machine-readable outputs are under
`results/full_393_workflows_four_path_v2/`; its historical `audit.json` passed
the then-current metadata gate but is superseded by the transport invalidation.
`artifact_hashes.json` pins every originally emitted artifact. The
older `results/full_393_workflows/` directory is retained only as superseded
diagnostic evidence because it reused protection bandwidths as recovery
bandwidths.

The checked-in invalidated template is deliberately not runnable:

```bash
uv run python AegisKV/experiments/real_weka_full_design/run.py
```

This command fails before creating an output directory or loading
the trace while `config.json` remains
`INVALIDATED_TCP_TRANSPORT`. Do not change that status by hand. After the
replacement Mooncake campaign produces a schema-v2 V100S profile, create a new
immutable config through the independent importer:

```bash
uv run python \
  AegisKV/experiments/real_weka_full_design/import_mooncake_v100_profile.py \
  --template AegisKV/experiments/real_weka_full_design/config.json \
  --physical-profile /path/to/v100/physical-profile.json \
  --source-manifest /path/to/v100/source-manifest.json \
  --source-verification /path/to/v100/source-verification.json \
  --pilot-runtime-gate /path/to/v100/dp4-pilot-runtime-gate.json \
  --aimd-pressure-calibration \
    /path/to/v100/dp4-aimd-pressure-calibration.json \
  --context-audit \
    AegisKV/experiments/real_weka_context_scale/context-audit-20260823.json \
  --selected-scale-label \
    scale-0.004-calibration-aligned-primary-candidate \
  --output-config /path/to/new-v100-sglsim-config.json \
  --output-manifest /path/to/new-v100-sglsim-import-manifest.json
```

The importer independently re-hashes the Mooncake profile, source manifest,
source verification, configuration manifest and its PASS verification,
four-condition pilot runtime gate, all four condition runtime profiles and
their underlying artifacts, context preregistration, and the profile's
calibration inputs. The pilot gate must bind the same campaign, source,
configuration, verification, and physical-profile digests as the imported
calibration. It requires four
distinct physical rates (source-to-Peer-HBM protection, source-to-Store-DRAM
protection, Peer-HBM recovery, and Store-DRAM recovery), the recompute fit, the
four-device V100S peer-access matrix, the pinned intra-enabled engine, and the
exact Qwen3-8B FP16 KV ABI. The imported primary workload applies the
preregistered 0.004 token projection, whose maximum 3,987-token context is
inside the 4,096-token physical calibration range. The primary simulation also
requires the physical TP=1, PP=1, system-DP=4 layout exactly; a DP=8 or PP=2
projection fails closed until that layout has its own profile. The result
gate must also exercise all four independent residency outcomes, the
HBM-reject-to-independent-DRAM path, independent tier acknowledgements,
HBM-only direct reclaim without a DRAM prerequisite, both physical recovery
paths, and zero target-owned/HBM-to-DRAM/destage events. The result
remains a metadata-only simulation calibration and never upgrades into
physical end-to-end evidence. With the optional pressure artifact, the
same-layout replay may quantify simulated AIMD pressure behavior; without it,
the active-source-request proxy keeps the replay diagnostic-only. Neither path
authorizes an end-to-end paper performance claim. The complete 393-workflow
entry requires the hash-bound pressure artifact; only limited development
slices may retain the disclosed proxy.

### Current V100S-DP4 v20-r2 source/control boundary

The unique current future-producer authority is the isolated root
`/home/comp/zhanzhao/aegiskv-execution/qwen3-v100-dp4-20260825-v20-r2`,
campaign `qwen3-v100-dp4-20260825-v20-r2`, hardware
`V100S-1node-TP1-PP1-systemDP4`, and design
`source_owned_independent_hbm_dram_v2`. SGLSim re-hashed the following static
artifacts locally on 2026-08-25:

- `isolated-source-staging-20260825T1300HKT-v20-r2.json` has SHA-256
  `05c5c4fdbd6831e1fb57179ce9de0465a778b6142f4c21e707a3138dd14e8858`,
  status `STAGED_SOURCE_ONLY_NO_JOB_SUBMITTED`, 228/228 affected tests, and
  empty job, submission, scheduler-query, and scheduler-mutation evidence;
- `source-manifest-isolated-v20.json` has SHA-256
  `5b22af168add23ace27df873cf985b2bf29d9b05e5bab8b53d9224a97a05508f`,
  exactly 108 entries, and entries digest
  `bf2f98fa2a0de1616aae9d43856c6c2f76397ebb4c9c646467d3f3d7da2fb1b6`;
- `source-verification-isolated-v20.json` has SHA-256
  `3ed051d354c8a9a2b07d20ae3e1c26104c641aeb7e090e5b02ae2c372919dba1`
  and reports PASS 108/108;
- `source-activation-isolated-v20.json` has SHA-256
  `ea3f7a7cfc88bceaa1e72e2e1f0db158e0686a91a2fe1f8541a47cdce4040a9a`.
  Its seven activation checks are true, including explicit supersession of
  queued-v11 source SHA `aad3549...`; it retains wire values `BYPASS=0`,
  `ADMIT_DRAM=1`, `INVALID=2`, and `ADMIT_HBM=3`; and
- `clean-preflight-submission-gate-v20.json` has SHA-256
  `f0b0cdc11f30db3e93a32f214c2f20af0e12c4387711a164c72710ab4ec6550d`.
  It independently re-hashes all 12 lineage references and authorizes at most
  one 600-second exclusive V100S-DP4 clean preflight. Model execution, config,
  physical calibration, mechanism, formal-scale, performance, and paper-ready
  authorization remain false.

The current clean-preflight contract fails closed on CPU-only foreign Slurm
jobs, foreign GPU processes, any topology other than four V100S SM70 devices,
a worker without a compatible Peer-HBM target, copy-integrity failure, or a
mismatch in the SM70 vLLM/PyTorch/CUDA/Mooncake/Qwen3-8B identity. Producer
tests for this source/control layer are 70/70 PASS. These are contract tests,
not observed CUDA, model, KV-transfer, or calibration evidence.

The singleton V100 monitor remains in an independent session with `/tmp`
logging. After the home-quota relief described below, its 13:39 HKT control
setting returned from the temporary read-only `QOS_LIMIT=1` fence to the normal
`QOS_LIMIT=10`. It still records `WAIT` at submitted count 10/10, with no r2
V100 job or submission record. Its source and clean-gate hashes are unchanged;
it never auto-cancels and has no downstream authority. Therefore r2 remains
current source/control readiness only and cannot enter SGLSim replay.

The 13:12 HKT read-only queue snapshot also identifies definite stale occupancy
inside that 10-job submission count: old-v11 job 131876 is
`PENDING/DependencyNeverSatisfied` at zero runtime because its `afterok:131875`
predecessor failed; 131877--131878 remain dependency-blocked and 131879 remains
held, also at zero runtime. None was cancelled or modified. These old jobs are
queue diagnostics only and cannot become r2 producer evidence.

The prior v20-r1 root is a superseded source-only diagnostic. The older v11
source and Slurm chain are also superseded diagnostics and cannot be joined to
r2 or consumed as current physical evidence.

#### Superseded V100 v11/v20 staging diagnostic

The historical consumer audit tracked the staged Mooncake V100 postprocessor
contract recorded by
`/home/comp/zhanzhao/Mooncake/aegiskv/campaigns/qwen3-v100-dp4-20260823-v11/postprocess/v20-current-design-producer-staging-audit-20260824.json`.
That audit reports 50/50 postprocessor tests, Python compilation, and
all 13 listed producer-file hashes passing. Its historical-v11 negative
activation probe checks 108 pinned entries
and fails as expected because that source freeze has not activated v20; its
source-manifest SHA-256 remains
`aad3549b40ec9bd8543d94367fba6d0a4a498272c400be06885255f5e903e9da`.
This is superseded static staging evidence only.

The same audit records a temporary-directory diagnostic in which v12 through
v19 apply in order (8/8), 143/143 affected runtime tests pass, and a temporary
109-entry manifest passes source activation. That temporary source/config tree
is deliberately non-consumable and is not runtime, physical, or performance
evidence.

#### Unchanged V100 schema-v2 consumer boundary

The producer contract includes a scheduler-read-only four-worker V100S vLLM
V2 model-execution probe and an immutable config-manifest/rebuild-verification
producer. The consumer independently re-hashes and validates the resulting
source activation, four unique rank/device identities, shared model revision,
model-config SHA-256, vLLM/PyTorch/CUDA identity, exact V100S-DP4 config input,
and its four-condition set. Producer source activation must also prove the
runtime wire enum `BYPASS=0`, `ADMIT_DRAM=1`, `INVALID=2`, and `ADMIT_HBM=3`.
SGLSim does not parse raw runtime decision integers: it consumes only the
producer-normalized named counters and PASS mechanism checks, so it never
interprets decision `0` as HBM admission.

The physical-profile producer now also invokes the config-manifest producer's
deterministic `verify_manifest` at consumption time and requires that rebuild
to equal the supplied PASS verification exactly. A forged outer PASS after
model-probe rank/device drift therefore fails closed. Recompute calibration
must use the same model revision pinned by the config manifest and four-worker
probe. These are producer-side hardenings; they do not change the profile
schema or authorize SGLSim replay.

The schema-v2 physical profile and every condition profile must share one
hash-verified `source_manifest`, `config_manifest`, PASS
`config_verification`, and `physical_profile`; only each condition's
`strict_run_audit` is local. The gate must establish clean telemetry for all
four GPUs, V100 PCIe-P2P Peer-HBM traffic, source-owned direct DRAM traffic,
independent acknowledgements, separate physical Peer-HBM and DRAM recovery,
and HBM-only reclaim without a DRAM prerequisite. Peer-HBM and DRAM protection
may overlap or execute separately, so timing overlap is not a correctness
gate. A negative recompute fixed overhead is invalid.

The old gate evidence class and old overlap check fail closed. Although the
probe and manifest *producer code* now exists, a usable V100 profile still
requires safe staged-source activation, actual four-worker probe output, an
actual config input/manifest/PASS rebuild verification, fresh calibration, and
four fresh physical condition runs. Until those artifacts exist, replay and
figure generation remain prohibited.

### Recorded A100-DP2 lineage extension (not consumed)

The Mooncake A100-DP2 static producer contract additionally requires a PASS
`a100_dp2_config_manifest_verification`. Its schema-v2 physical profile exposes
top-level `config_manifest` and `config_verification` artifact records. Each of
the four condition profiles records this exact `input_lineage`:

- `source_manifest`
- `config_manifest`
- `config_verification`
- `physical_profile`
- condition-local `strict_run_audit`

The Mooncake mechanism gate requires the first four records to be identical
across all four conditions; only `strict_run_audit` is condition-local. The
model probe bound by that configuration lineage records and checks the model
snapshot revision, model `config.json` SHA-256, and vLLM, PyTorch, and CUDA
versions across both workers.

Lineage must also propagate through both physical calibration branches. The
recompute `a100-recompute-inputs.json` binds the same `source_manifest`,
`config_manifest`, and PASS `config_verification`. Each of the three transfer
treatments has exactly two independent, condition-local
`a100-transfer-inputs.json` records, for six physical Slurm runs in total; all
six bind that same source, configuration, runtime, and verification tuple. The
transfer audit additionally checks that its recompute gate uses exactly the
same `config_manifest` and `config_verification`; a cross-branch mismatch
therefore fails closed.

#### Current A100-DP2 v44-r31 source/control boundary

The current source/control authority for a future A100 bridge is the isolated
execution root
`/home/comp/zhanzhao/aegiskv-execution/qwen3-a100-dp2-20260824-v1-v44-r31`.
The design remains `source_owned_independent_hbm_dram_v2`. SGLSim re-hashed
the following source/control artifacts locally on 2026-08-25:

- `isolated-source-staging-20260825T122400HKT-r31.json` has SHA-256
  `24e0f9a096e05626aff3709dde4dfa0df99c5a664e1b92d83371c1de9343de32`,
  status `STAGED_SOURCE_ONLY_NO_JOB_SUBMITTED`, and explicitly disclaims
  clean-GPU, model, transfer, calibration, mechanism, and performance evidence;
- `source-manifest-isolated-v19.json` has SHA-256
  `6604e28142b33cbb7a1dd220fb427cfd5efc73bc78486f52f2ccd7ea79891e23`,
  contains exactly 108 entries, and records entries SHA-256
  `bf2f98fa2a0de1616aae9d43856c6c2f76397ebb4c9c646467d3f3d7da2fb1b6`;
- `source-verification-isolated-v19.json` has SHA-256
  `3e45b40b871a464bfec3ef4d7a871f87ca5f4f01609c6092eb683e1527b0d6d9`
  and reports PASS 108/108;
- `source-activation-isolated-v19.json` has SHA-256
  `199a59213abfabc76ee7184e7eaaeaf237ae53943801b106d97a4b854748159d`
  and reports PASS for the current design with `jobs_submitted=[]`;
- the byte-identical live/isolated activation contracts have SHA-256
  `7166e448ba3e17ac9e1ca1b57cf16ec2b7f7570f0026a75c441793cb143cc119`;
- `static-first-run-readiness-v106-r31.json` is PASS 16/16 with SHA-256
  `81af8478a7972707575b5538dde9d439f1e6a96822227d4e31da9d5ef48a533a`;
- `manifest-v106-r31.json` has status `STATIC_READY_RUNTIME_UNVERIFIED` and
  SHA-256
  `d0c290f75030431d7875ff9e60e483e325a532758388ce969ed6ff010735183e`;
- `clean-preflight-submission-gate-v45.json` is PASS with SHA-256
  `f71be52cb7451401eb9d775dd702943af43d0663aa726d2bf9b7dcf82afe653b`;
  it is a Mooncake clean-only gate, not an SGLSim consumer preflight; and
- `runtime-chain-control-rebinding-v106-r31.json` is PASS with SHA-256
  `8e2305f9c603ff751fbb42006aa97ddd23b02f0fe6818ca82a9dd03b42bdbd63`.
  It re-hashes 15 controls and five deterministic inputs and records no
  scheduler query or mutation, CUDA/model execution, or runtime/physical result.
- `source-verification-r31-pre-monitor.json` has SHA-256
  `d1a2fc870e5ead7ca670b59e4f6068a4cabefdc73ab091ae7f06fa398126edc0`
  and independently reports PASS 108/108 immediately before monitor startup.

The producer reports 300/300 current campaign tests after excluding three
historical patch-replay tests that are inapplicable because those patches are
already integrated. A separate live physical-chain regression is 47/47 PASS.
That source correction makes `audit_registered_gpu_copy.py` accept the current
v47/v46/v48 evidence classes while independently checking the v47
`auditor_revision`. It also makes `hbm.py` and `peer_hbm_runtime_probe.py`
explicitly and idempotently unregister source regions before the source tensor
lifetime ends. These are source/test facts only, not an observed physical copy.

The r31 clean submission record has SHA-256
`b2314cd372dcea58d34b4a085554dfa584c8462b950a2db7273a5992febd7929`
and identifies job 139550 as `QUEUED_NO_RUNTIME`. Its current supplied state is
`PENDING/ReqNodeNotAvail`, elapsed zero, fixed to gpu26, with no assigned node or
GPU. At 13:12 HKT gpu25 was idle with both A100s free, but neither 139550 nor the
r28 model job was rebound or resubmitted.

The original 16-entry monitor PID table has SHA-256
`c14a889bf130ffaa865a332e6524ab511c53ff24ea8185de043735c3ae700458`,
but it is now a pre-quota-failure process snapshot. At 13:01 HKT, the local
v45/v46 monitor processes exited before any scheduler mutation because home was
at its 600/600-GB quota, `TMPDIR` pointed into home, and Bash could not create a
here-document temporary file. Job 139550 remained pending at zero runtime; it
was not run or cancelled, and no replacement job was submitted.

The quota root cause was recoverably relieved at 13:39 HKT. The untracked
`build-sglang-profiler` trees under the live checkout and its only L40S v23-r1
hard-link lineage were each moved intact to
`/tmp/aegiskv-home-quota-relief-20260825`: each tree contains 195 files and
3,478,681,968 bytes (about 3.3 GB), leaving about 3.2 GiB of home headroom. No
source, model, manifest, raw result, or Slurm record was deleted.

The next watchdog scan at 13:42 HKT confirmed `root_count=3,total=19`:
r31=16, r28=2, and V100=1. The r31 v45/v46 and r28 v46/v47 monitors all
rediscovered their original zero-runtime jobs; no new job was submitted. The
r31 sessions retain `TMPDIR`, `TMP`, and `TEMP` under `/tmp`, and the live
future-starter still fails closed if that location is not writable. This is
control-plane recovery only and changes no r31 frozen source or V100 r2 producer
hash.

The live observe-only v63 watchdog now also classifies a monitor launched
directly from the mutable Mooncake checkout as a superseded root. Its current
rehash-bound gate has SHA-256
`dad310245e96a94f7e5a806e9ee8689210398d096552a2549282ae19b20bd3ff`,
with supplied validation 5/5 PASS. It found seven additional approximately
12-hour-old superseded monitors under the mutable checkout; together with PID
3231236, all eight were matched by exact PID plus command line and sent SIGTERM.
Only local monitor shells were affected; no Slurm job or foreign process was
modified. The subsequent paper-status validation is 7/7 PASS. The watchdog
remains observe-only with `action=no_auto_cancel`, and none of these changes is
frozen runtime, physical, or performance evidence.

The immediately preceding r30 clean job 139521 is resource-contaminated
diagnostic evidence only. It ran for 14 seconds on gpu27, observed external PID
1311059 using 21,950 MiB before the clean preflight could proceed, then ended
`FAILED` and released the allocation. It produced no model, calibration,
mechanism, or performance result. That attempt also exposed a v46 monitor bug
that treated a transient `SUSPENDED` state as terminal; the current control
code fixes that classification without turning r30 into consumable evidence.

Older lineages remain isolated. In particular, r28 model job 139406 remains
`PENDING/Resources` at zero runtime, fixed to gpu26 with no assigned GPU. Its
v46 and v47 monitors are both alive again and have rediscovered that same job;
the already-disabled downstream monitors remain stopped. This r28 control state
cannot enter the r31 result chain. The r26 model-only diagnostic and r27 pending
diagnostic likewise cannot supply or repair r31. Jobs 131876--131879, 139406,
and 139550 were not cancelled or rebound. SGLSim performed no scheduler mutation
or Slurm action while recording this boundary.

This r31 source/control and queued-clean boundary is not a consumable physical
lineage. There is still no same-r31 model/config/registered-copy PASS,
recompute, six-run transfer series, schema-v2 physical profile,
four-condition mechanism gate/current strict-audit set, independent SGLSim
consumer preflight, paired effect, replay, or performance result. A scale
campaign requirement for r31 therefore remains `BLOCKED_MISSING_LINEAGES`.

#### Superseded A100-DP2 v44-r28 diagnostic boundary

The superseded r28 source/control root is
`/home/comp/zhanzhao/aegiskv-execution/qwen3-a100-dp2-20260824-v1-v44-r28`.
The design remains `source_owned_independent_hbm_dram_v2`. SGLSim re-hashed
the following producer artifacts locally on 2026-08-25:

- `isolated-source-staging-20260825T103200HKT-r28.json` has SHA-256
  `56e6803df9499444780a5f1f4db2ff949da1bb5634b2cd8b5148bdd2934ddc42`;
- `source-manifest-isolated-v19.json` has SHA-256
  `7464917eaa5eaede4166e0e7be8c6803489b40c7c331c6e7b9a836bc793d6da3`,
  contains exactly 108 entries, and records canonical entries SHA-256
  `622a54b6da81775bf52865483c950ee2861444842011aeef1fb4f44d45aa6ff2`;
- `source-verification-isolated-v19.json` has SHA-256
  `dac9cafce05f8985970410721310b595ffd23ebf0e640d530c47005061f8ed45`
  and reports PASS 108/108;
- `source-activation-isolated-v19.json` has SHA-256
  `f094166821a4877e888db783afa375130d781c461078b12e5e829a89f5e90cab`;
- the byte-identical activation contracts have SHA-256
  `a8df2b63d9d9c83defb28ddcef742139cecff4f361838c61d3049a7d91f74156`;
- `static-first-run-readiness-v104-r28.json` is PASS 16/16 with SHA-256
  `6aac64572731c08bd7e69acf47ee08c5a7b13f6013243805d485e906890a80a4`;
- `manifest-v104-r28.json` has status `STATIC_READY_RUNTIME_UNVERIFIED` and
  SHA-256
  `c97e7a0e26386397b5a09ab2203b109134e9cae598640da7e9c77c02b9aec48b`;
- `clean-preflight-submission-gate-v104-r28.json` is PASS with SHA-256
  `8474ab1fac862df63af519c849d2d63f72cb8d7637c6e99b81ceba58714741a8`;
  it is byte-identical to the default `clean-preflight-submission-gate-v45.json`,
  is not pinned to a particular node, and remains a Mooncake clean-only gate,
  not an SGLSim consumer preflight; and
- `runtime-chain-control-rebinding-v105-r28.json` is PASS 15/15 with SHA-256
  `02c16c6b0623a5e75d46125a73ce84f98aadbf83ebc3cccc727b8777c51afa59`.

The producer reports 266/266 campaign tests and a fresh 108/108 source rehash.
At 2026-08-25 11:00 HKT, `start_runtime_chain_monitors_v70.sh` started all 16
fail-closed local monitors. The immutable PID table
`runtime-chain-monitor-pids-v70.tsv` has SHA-256
`68456c10401aa636f0180103d952070fb59d6a4959279b649921d85099578b87`;
SGLSim's earlier read-only process check found all 16 listed monitor PIDs
alive. The runtime boundary has since advanced within the same r28 lineage.
Mooncake job 139379 completed on gpu26 with `COMPLETED/0:0` after 29 seconds.
The PASS clean audit is
`a100-dp2-clean-preflight-139379-20260825T032807Z/clean-gpu-preflight-audit.json`
with SHA-256
`1d111a47ea1bbf3798b0b42b6bcb16b7852c53684970299c0571dd087e93f8cb`;
it verifies only the clean A100-DP2 resource/runtime preflight. The subsequent
model gate and submission records have SHA-256
`718b29a4538c88960b0fcfc9073250bdcd084d9db74fa729d34a368ea2106bf4`
and
`75d86cd7a8663952f4efd8dfd60e691ccd4e351e6802ac9ffebc4201d8d19455`,
respectively. They submitted exactly one model job, 139406, at 11:28:55 HKT;
the latest supplied snapshot is `PENDING/Resources`, elapsed zero, with no
assigned node or GPU. The clean one-shot monitor exited normally; the 15
broken downstream r28 monitors were later stopped after r28 was superseded.

A later local process census sent `SIGTERM` to 136 superseded r7,
r13--r18, and r23--r26 monitor shells; it did not mutate Slurm state.
The then-retained monitor population was r27=16 plus r28=15. That census is a
historical snapshot, not the current monitor boundary recorded in the r31
section above. The read-only v63 watcher never sends process signals or
scheduler cancellation commands.

This is a clean-PASS plus queued-zero-runtime model boundary, not a consumable
physical lineage. r28 still has no model PASS, registered-copy, recompute,
six-run transfer, physical profile, four-condition mechanism/recovery, paired
effect, or performance artifact. Its clean-only gate and PASS clean audit are
not the independent SGLSim A100 consumer preflight. A scale-campaign
requirement for r28 therefore remains `BLOCKED_MISSING_LINEAGES`; assigning an
r26 or r27 binding instead is a cross-lineage error, not a readiness shortcut.

The r26 execution root is retained as a separate diagnostic-only lineage. Its
job 139246 completed `0:0` in 133 seconds and its v47 model-attempt audit is
PASS for Qwen3-8B FP16, two A100 GPUs, TP=1/PP=1/system-DP=2, and two vLLM V2
workers. It records 55 complete telemetry frames, per-device peak utilization
of 100%, peak combined framebuffer use of 57,358 MiB, and 58 co-residency
samples. This proves only model/runtime/telemetry viability within r26. No
registered-copy, calibration, mechanism, or performance result exists there,
and no r26 artifact may be joined to or substituted for r28.

The r27 clean job 139272 remains a zero-runtime `PENDING/Resources` diagnostic
bound to gpu27 and is not an allocated-GPU result. Its source includes the
registered-copy gate correction but predates the current calibration-gate
correction, so it is not a complete current lineage and cannot satisfy the
A100 consumer. The new r28 jobs above do not repair or consume r27. SGLSim
performed no scheduler mutation or Slurm action while recording this boundary.

The current live/r31 source contract extends each Aegis strict audit with
`measurements.aegiskv_data_plane.aimd`. Its deployment config freezes
`total_hbm_bytes`, interval, memory-pressure threshold, physical local-stall
threshold, alpha bounds, multiplicative-decrease factor, additive-increase
step, and stable-window threshold. The strict block records an exact per-pool
trajectory plus aggregate alpha, transition, memory/local-pressure, quota,
and reclaim summaries. The strict auditor re-derives pressure from CUDA and
target-local operation samples and replays the target-local decrease/increase/
stable-window/quota law. Pool cardinality comes from the frozen Router/hardware
topology, so this DP2 path requires two pools rather than a hard-coded four.
`analyze_campaign.py` cross-checks strict summaries against same-run raw events;
`review_pilot.py` requires audited AIMD for Aegis conditions and zero AIMD
activity for no-protection controls.

The same-run `protection_effectiveness`, copy-to-hit, failure-local WCT, and
direct whole-workload failed-run WCT rules remain in r31. They are static
source contracts, not observed results. The Mooncake clean-preflight gate is
**not** the independent `sglsim_a100_dp2_consumer_preflight_v2` artifact and
cannot substitute for it. The r31 clean job is queued with zero runtime, so
there is still no same-r31-lineage clean PASS, model/config/registered-copy
PASS, recompute or six-run transfer calibration, schema-v2
physical profile, four-condition gate/profile/condition-local strict-audit
set, SGLSim consumer preflight, paired effect, physical KV movement, matched
replay, figure, or performance result.

Superseded r12/r13/r28/r30 and other intermediate source/control lineages
cannot be joined with r31. The r26 model diagnostic and r27 pending job remain
separate for the reasons above. Existing r7 queued/held jobs also remain a
separate historical lineage and must not be joined with r31 artifacts. The r7
queued-control lineage uses
the isolated execution root
`/home/comp/zhanzhao/aegiskv-execution/qwen3-a100-dp2-20260824-v1-v44-r7`
with `activation_revision=v44-affine-six-run-calibration`. Its historical
source/control artifacts are:

- `source-manifest-isolated-v19.json` has SHA-256
  `3445e6a19e4198844f407507deb86d101e78b859211dfbc72ae9c61ee0da2680`,
  contains exactly 108 entries, and records canonical entries SHA-256
  `97e476833f82c7ec656ac85cc5d0da47dd8b6134c7a38f6f4fb893ebb1f61500`;
- `source-verification-isolated-v19.json` has SHA-256
  `41d987e7baf0b709d82b3de6ffd4c418c0bb2395fd0c7872ce0bbdb2bfe0f834`
  and reports PASS 108/108 with no failures;
- `source-activation-isolated-v19.json` has SHA-256
  `0a6331784cc141b0075efc2e20b744bb1762b5adb97b835c8c97e7a5c6f42ba4`
  and reports PASS for `source_owned_independent_hbm_dram_v2`, with no
  submitted jobs;
- `isolated-source-staging-20260825T004004HKT-r7.json` has SHA-256
  `94bfeeb68f80935a3ba4ddde96040f8d6b548d4e621e2b6e62aab179f78a9673`
  and records 162/162 affected isolated tests passing;
- the byte-identical live and isolated `activation-contract.json` has SHA-256
  `f9df29d6334986a06ef573a9158b3b1b5c2a4ebebbbd3f412c1cab7834c35eda`,
  status `ISOLATED_SOURCE_ACTIVATED_RUNTIME_UNVERIFIED`, and
  `submission_authorized=false`;
- `static-first-run-readiness-20260825T004012HKT-v44-r7.json` has SHA-256
  `ed85220288af696747e699cd2aeb544879700ba170d9aa04fdbe34f3046746dc`
  and reports 15/15 static checks passing without Slurm, GPU, CUDA, runtime,
  physical, or performance work; and
- the current campaign `manifest.json` has SHA-256
  `d74cb4d67f23e971a92de411d2c567a9cbd7454c0d43b16a8638c04a1f55f6cf`
  and immutably records `PASS_535_OF_535`. The subsequently expanded complete
  applicable campaign-and-core regression through the live-control-only v63
  watchdog now records 584 passed and 1 skipped after adding two read-only
  resource-watchdog tests. These
  control-only additions do not retroactively mutate the v45 gate or its bound
  manifest and do not create runtime, physical, mechanism, or performance
  evidence. The sole skip is due to vLLM being absent from the lightweight
  control environment; the isolated runtime identity remains separately gated
  and contains the pinned vLLM installation.

Because the home filesystem was at 597/600 GB, the r7 runtime assets use
explicit hard-link deduplication from the current live runtime. Staging checked
every regular asset by identical device/inode identity and every symlink by its
target. The activation contract requires a complete runtime re-hash before any
run, so drift in either hard-linked view fails closed. This is storage
provenance only, not runtime evidence.

The v44 A100 transfer contract remains exactly three treatments
(`aegiskv`, `aegiskv-no-hbm`, and `aegiskv-hbm-reject`) times two independent
physical Slurm runs, for six runs total. Each run must expose one true
run-derived `calibration_stream_id` and actual issue-order
`calibration_transfer_ordinal` values. The resulting profile must represent
all four physical paths with separate affine per-transfer setup,
first-stream warm-up, and steady-state bytes/ms terms, while binding one exact
source/config/runtime lineage.

The r7/v45 clean-preflight launcher is now `#SBATCH --exclusive` with a
600-second hard limit. Before its first `nvidia-smi`, CUDA, Mooncake, or model
operation, it reruns the static-first-run auditor and requires the exact
15-check result. It re-hashes the 108-entry source manifest, source
verification and activation, and all bound runtime launchers, auditors, and
builders. The resulting clean audit must contain the additional hard gate
`pre_run_source_runtime_identity`; a mismatch stops the run before any GPU or
runtime observation.

The immutable `clean-preflight-submission-gate-v45.json` has SHA-256
`e1e740a8a7c63807c20c741cc8d905c0f7fd02bbd81e68d0afe472f547813fac`,
status `PASS`, and `jobs_submitted=[]`. It authorizes at most one exclusive
clean preflight (`maximum_inflight_clean_preflights=1`) under exact Slurm
comment `aegis-a100-v45-clean-e1e740a8a7c6`. Model execution, physical
calibration, mechanism, formal-scale, and paper-ready authorizations remain
false.

The user unit `aegis-a100-v45-clean-monitor.service` started at
2026-08-25 00:48:18 HKT and is active/running. It first recovers any
exact-comment existing job, then applies the QOS-count fence. The current log
contains only `WAIT qos_submitted_count=10`, so it has not called `sbatch` and
no new job exists. If a submitted run reaches at least 180 seconds while both
its logs and artifacts remain unchanged for at least 180 seconds, the monitor
only emits `ALERT ... action=no_auto_cancel`; it never automatically invokes
`scancel`.

The superseded v26 model gate and monitor hard-coded the old v20-r4 root and
clean job 135943, so they cannot continue a dynamically submitted r7 clean
run. The live control-only v46 continuation replaces that control path without
changing any of the frozen 108 source-manifest entries. Its immutable
`model-continuation-control-gate-v46.json` has SHA-256
`6180bbf32b181d5b6d0a0a64203b434c67f7c8be7faf75bc658ca3c62ac96b5f`,
status `PASS`, `maximum_inflight_model_probes=1`, and
`automatic_cancellation_authorized=false`. It binds these control artifacts:

- `collect_runtime_identity_v46.py` SHA-256
  `c87c9e78c520775c7539d1fd99669d627cfc55fe3ac0d1d385641b310f8f96f2`;
- `build_model_submission_gate_v46.py` SHA-256
  `7c52a537238e7853394f5acbf138c221650e9bef1873a81620537ea3fbee0752`;
- `monitor_v46_model_submission.sh` SHA-256
  `537937dd5dc2f72fbeaef8438e00e8213ec35737cf895efdf8dd87f9cb584f14`;
- `test_model_continuation_v46.py` SHA-256
  `252fe1d40b276c803ffce43e8f4c74dfd040c0ae4791e4ae4137e9433187c4fa`;
  and
- the r7 v45 clean-submission gate, source activation, model launcher and
  coordinator, plus the static runtime-identity diagnostic described below.

The control-only `runtime-identity-static-diagnostic-v46.json` has SHA-256
`806aee276c82ce1437a2e9039d362f07a9ccd83e3661f88bee7db40a76e52762`
and records vLLM `1.2.2+aegiskv.sm89`, PyTorch `2.10.0+cu128`, and CUDA `12.8`.
Its Python executable, vLLM package, V2 model runner, Torch package, Torch
native extension, and Mooncake engine are all rooted in the same r7 execution
tree and hash-verified. Collection imported the pinned packages but did not
initialize CUDA, load the model, move KV, or measure performance; the artifact
therefore has `gpu_or_model_execution_performed=false` and
`runtime_or_physical_result=false`.

After v45 writes an immutable submission record, v46 reads its actual
`clean_job_id` rather than a preselected job number. It accepts only Slurm
`COMPLETED/0:0`, elapsed time from 1 through 600 seconds, two A100s, fresh
108/108 source verification, a revalidated runtime identity, and exactly these
eight clean gates: `bidirectional_cuda_p2p_and_integrity`,
`mooncake_cuda_engine_import`, `pinned_vllm_v2_runtime`,
`pre_run_source_runtime_identity`, `qwen3_8b_snapshot_available`,
`slurm_node_exclusive`, `two_a100_80gb_sm80_devices`, and
`zero_foreign_gpu_processes`. Only then may it submit at most one exclusive
model probe. QOS count 10 or greater remains a wait state; a running model job
with no log/artifact update for 180 seconds only emits
`action=no_auto_cancel`, with no `scancel` path.

The user unit `aegis-a100-v46-model-monitor.service` is active/running and its
current log contains only `WAIT clean_submission=absent`. The v45 monitor
remains active/running with `WAIT qos_submitted_count=10`. Thus neither stage
has submitted a new job.

The earlier v27 post-model auditor always loaded the v25 runtime-binding
verifier, so it could not validate a dynamic r7/v46 binding. The control-only
v47 audit layer fixes that continuation without changing the frozen source
identity. Its immutable `model-audit-control-gate-v47.json` has SHA-256
`be5d9ba5a7bfe2218b31874dc70bd0320e4c435b024ed2e6abc2f119ed146379`,
status `PASS`, `audit_only=true`, and
`registered_copy_submission_authorized=false`. It hash-binds:

- `audit_runtime_binding_v46.py` SHA-256
  `bd921c0b5c4936dd22382f2b4ff8fc49652878a3a65b5256480c972e50bd0f7e`;
- `audit_model_execution_attempt_v47.py` SHA-256
  `1582f92b6782d34ad7908094769bf822df44b401432f04ec66c35ec8a27d08ed`;
- `monitor_v47_model_audit.sh` SHA-256
  `ccaf473e189f2c1a6b64e25c3aea89ea3ecc728fbff377cf69b42dcf65395c3e`;
- `test_model_audit_v47.py` SHA-256
  `f165abec6fa4c715572b5e5e195462bf46cea09227bdf6f03d7b4320e60ca3f4`;
  and
- the v46 continuation gate and model-gate builder plus the base v27 model
  auditor.

v47 accepts exactly one v46 model-submission record and re-hashes its model
gate, runtime binding, and source activation before observing terminal state.
After Slurm `COMPLETED`, the audit requires two distinct Qwen3-8B/vLLM V2
workers and devices, matching model revision/config SHA and runtime identity,
pre/post source verification at 108/108, exclusive Slurm/co-residency evidence,
multi-frame telemetry from both GPUs with positive utilization and framebuffer
activity per device, zero leaked GPU processes, complete checksums, final
`sacct`, and Slurm output. A PASS authorizes only construction of one later
registered-copy submission gate from the same lineage; it neither submits the
copy nor authorizes calibration, mechanism, performance, formal, or
paper-ready claims.

The user unit `aegis-a100-v47-model-audit.service` is active/running and its
current log contains only `WAIT model_submission=absent`. Its monitor contains
no `sbatch`, `scancel`, `scontrol`, or `srun`, so it cannot submit or cancel a
model, copy, or calibration job. The v45 and v46 monitors remain active with
their prior QOS and clean-submission wait states.

The frozen r7 `registered_gpu_copy_probe.sh` still required the superseded
v20-r4 model audit, v25 runtime binding, and v28 submission gate. Those frozen
108 source entries were not changed. Instead,
`build_registered_copy_launcher_v48.py` performs exactly four verified,
single-occurrence replacements: the v47 model audit class, v46 runtime-binding
class, v48 registered-copy gate class, and the launcher's self-recursive
overlay path. The generated extra overlay
`registered_gpu_copy_probe_v48.sh` has SHA-256
`02add8d3f848d0a0f4e9c2f1b92dab6f89985fe00c846f0f6e8d705e82f0ffb9`;
the old evidence classes have zero remaining occurrences. The core source
manifest remains 108/108, while the overlay is independently hash-bound rather
than represented as a frozen-source mutation.

The immutable `registered-copy-control-gate-v48.json` has SHA-256
`ad189a7b6d6d1ea05e2a27764aca581c625f3d0455a4a1efbfe3ac42a06e3c61`,
status `PASS`, `maximum_inflight_registered_copy_probes=1`, and
`automatic_cancellation_authorized=false`. Its principal control files are:

- `build_registered_copy_launcher_v48.py` SHA-256
  `24e8246c8ce265e962b7dca3f7cd1150e3dfa45a254453d403f83b64281d8f46`;
- `build_registered_copy_submission_gate_v48.py` SHA-256
  `fa1d90598e226466eecd5c3d581bc6b7d847d9856d9f818756047632542e676b`;
- `monitor_v48_registered_copy.sh` SHA-256
  `2308ce58c9e730787c5d72b371d4fe20f5514a86daf12f77b841da74d75070e3`;
  and
- `test_registered_copy_v48.py` SHA-256
  `363adcfcfc2a228e8e4d48c5a4897ff17b61127126648b764a6422dec95e517e`.

Only after a PASS v47 model audit may v48 freshly verify the 108-entry source,
revalidate the v46 runtime binding, and join model/clean/source lineage before
queuing at most one exclusive, source-owned registered CUDA-P2P probe. A QOS
count of ten or more remains `WAIT`; 180 seconds without log/artifact progress
only emits `action=no_auto_cancel`, and no `scancel` path exists. The user unit
`aegis-a100-v48-registered-copy.service` is active/running and currently logs
only `WAIT model_submission=absent`.

The v49 audit wrapper reuses the complete v33 terminal-copy checks while
compatibly presenting the v48 submission class in memory; it does not relax
the checks themselves. The immutable
`registered-copy-audit-control-gate-v49.json` has SHA-256
`9fddbe80750a4c7f521c05d1ad437ae25d20335cdda983e166e7eba729297bc2`,
status `PASS`, `audit_only=true`, and
`physical_calibration_submission_authorized=false`. It binds:

- `audit_registered_copy_attempt_v49.py` SHA-256
  `94cf8166ebcc0292ffee03e46b64f3e58d50b0f937f76c3bebf5a51656c6a38d`;
- `monitor_v49_registered_copy_audit.sh` SHA-256
  `377b3d719a99cf7b3512d9ab75fcfd7919e906353734dedf404d925317ebc55d`;
  and
- `test_registered_copy_audit_v49.py` SHA-256
  `e5902f751e9117b6afcfbc44f9ca53e6ca501117e4740468d463a3942a309115`.

After a terminal copy attempt, v49 verifies complete checksums, final `sacct`,
the inner registered-copy audit, `transfer_ms`, effective GiB/s, and Slurm
completion markers. A PASS authorizes only construction of a later
physical-calibration gate and submits no calibration. The user unit
`aegis-a100-v49-copy-audit.service` is active/running and currently logs only
`WAIT copy_submission=absent`. Thus all v45--v49 services are active, but no
new clean, model, registered-copy, or calibration job has been submitted.

The v50 recompute-calibration control layer extends a PASS v49 copy audit into
at most one exclusive recompute calibration on the same registered-copy node.
Its immutable `recompute-calibration-control-gate-v50.json` has SHA-256
`2acb5aff2f2cd64ad6925a0fad6aa9707cf7a9a9c16541c4f2f6ae12734581f4`,
status `PASS`, `maximum_inflight_recompute_calibrations=1`,
`same_registered_copy_node_required=true`, and
`automatic_cancellation_authorized=false`. The generated r7 launcher overlay
`a100_recompute_profile_v50.sh` has SHA-256
`fec7db65fb8cf32062d6ac6a4f51a4e3e8e74e4134c5ea6332738763684712fc`.
Its principal control files are:

- `build_recompute_launcher_v50.py` SHA-256
  `f63c53b1f3defac5ff01981e6ac658b920d5c5e0b877be4ae53fb014387e112d`;
- `build_physical_calibration_submission_gate_v50.py` SHA-256
  `9b2e5455e0ffad3a41c375b34d48e82d1b9bf3e43eeb4254609cc7c3a7ee884b`;
- `monitor_v50_recompute_calibration.sh` SHA-256
  `52b9daf650d5b5a690c05761424940460d575c52b37013ca22f765f8bada2d93`;
  and
- `test_physical_calibration_v50.py` SHA-256
  `ad0ec67e56773fbd6a9c7eb536f6a322c329af530758bc4e277040f44ede186e`.

The audit-only v51 layer retains the complete v35 recompute checks while
compatibly accepting the v50 submission/gate identities. Its immutable
`recompute-audit-control-gate-v51.json` has SHA-256
`148d5116c9367b0c08fedb9bc5ab627e2230fd056ea05a5020717a53371f190d`,
status `PASS`, and `transfer_calibration_submission_authorized=false`. It
hash-binds `audit_recompute_attempt_v51.py` SHA-256
`c069c03510c311e002cee9f7cf821d75a356c9a5a6397a0682b8bdd2d2f11ddf`,
`monitor_v51_recompute_audit.sh` SHA-256
`f3fdcb955585630da249f123586a5f263f037d4651638d1242b3736d71c2bfa9`,
and `test_recompute_audit_v51.py` SHA-256
`99c3ffd58bfb450e66e373d9e175c1c25c4486b2f393850e28d138a499e85f3a`.
A PASS may construct the later transfer gate but does not itself submit a
transfer job.

The immutable `transfer-series-control-gate-v52.json` has SHA-256
`cf8ce7a84f8bdbd80d26af848614649a1d13f42c56b602c81ff0b2019b840d7f`,
status `PASS`, `maximum_inflight_transfer_calibrations=1`, and
`same_recompute_node_required=true`. It predeclares exactly three treatments
(`aegiskv`, `aegiskv-no-hbm`, and `aegiskv-hbm-reject`) with exactly two
independent repeats each. The six exclusive runs must execute sequentially on
the v51 recompute node and use exact queue-plus-accounting comment deduplication.
A QOS count of ten or more waits; 180 seconds without log/artifact progress
only emits `action=no_auto_cancel`.

The frozen `a100_transfer_calibration.sh` remains byte-identical at SHA-256
`dba801f4eabaa37dc2932cc952e3bea409cd97038fd3ad0b0714ac8dd7e22904`.
Rather than changing it, `build_recompute_compatibility_view_v52.py` SHA-256
`71d09876d311813bbb4a3f501cec50028addea0714b2f0ceec1877c15daafb77`
creates a hash-bound v51-to-v35 wire view that changes no measurements, fit,
node, or source lineage. `audit_transfer_attempt_v53.py` SHA-256
`3405db9629109dcb63e0bb53ea64de6acc2282252d22326cac673489772a83d9`
reruns the complete v37 terminal-transfer audit and then rechecks the original
v51 lineage. The v52 monitor and regression test have SHA-256 values
`f859b563d1f41118f3c900145c9cfd813327a47bcf6de1fa76475426a8ffba74`
and `e974c36f6e0c7d8b31234c00fd3abda90d7f8050215a41fef5f2731af7987b03`,
respectively.

The user units `aegis-a100-v50-recompute.service`,
`aegis-a100-v51-recompute-audit.service`, and
`aegis-a100-v52-transfer-series.service` are active/running. Their current
logs contain `WAIT upstream_submission=absent`,
`WAIT recompute_submission=absent`, and
`WAIT recompute_submission=absent`, respectively. Together with v45--v49,
these are active control monitors only; no recompute or transfer job has been
submitted and no v53 terminal audit exists.

The local-only v54 profile layer is guarded by
`physical-profile-control-gate-v54.json`, SHA-256
`7c0ef138cddfe591d54db7f2240ef49e6f9a9de78cc5185fb26ce98eaed7d04f`.
It will invoke the frozen schema-v2 profile builder only after
`transfer-calibration-series-ready-v53.json` proves exactly six PASS v53
attempts and their shared v51 lineage. Its auditor, monitor, and test have
SHA-256 values
`05603efc188d4360ed83f4d1e55fbccfdabd265ae4016143d133e504bcf4cd1e`,
`b720bcb0414de6dfab94d5fe04b0b1fa6568c5fc47df3d2127c2965446b26e5f`,
and `91a575297462b1ff3dc2f5b82b7fd158610d80118cc70da45ab2f2d3c7080c09`.
The gate is local postprocessing authorization only: it sets
`runtime_or_physical_result=false` and does not authorize a mechanism or
paired campaign.

The v55 `mechanism-preparation-control-gate-v55.json` has SHA-256
`30f0d13ef1615c9001469e4cfef066ccfc0243d882c8bc0db5cbcce3ee276a10`
and `workflow_count=60`. Only after the v54 profile/audit exists may it derive
the pre-workload physical AIMD local-stall threshold and deterministically
materialize the 60-workflow mechanism inputs. The v55 AIMD builder,
preparation builder, monitor, and test have SHA-256 values
`add5ac13b72e7b4d1b831f488045888e296383c3fc8d44f55addc7f97d1b6415`,
`309ca388f763caf81ff94dd49c2c7e4aa9fd7cad4c81509bf45f094586f72080`,
`a682caa505e3a96a2ea8a56bef0912366be3299e5bc8f0410353b13beeba9f80`,
and `c1f64f403a35b7d2b28c15de74f7077e1eef6b58ead2c606f2e3c1d3e13b9b6e`.
This stage is local materialization only and cannot use performance direction
or authorize a runtime condition.

The v56 `mechanism-gate-control-v56.json` has SHA-256
`01203386d96f33b0287acda0599db5d183edbab362314e077abd24f8b09cc6f2`.
Its builder, SHA-256
`5b48688a0cabdfef82ae0526f920e35a6bdc0b30b1c009c23fe16f0d89c7237f`,
creates three hash-bound v55-to-legacy-wire compatibility views solely so the
byte-identical frozen v40 planner and launcher can consume current inputs. It
does not change any measurement, fit, node, source, or physical-profile
lineage. The gate remains local planning only, fixes four conditions with at
most one in flight, and leaves mechanism submission unauthorized until the
v55 artifacts validate.

The immutable `four-condition-series-control-gate-v57.json` has SHA-256
`294061c86a3d7c1968c6f5c7562c75d7a1130b2e5efcc08cb712d695b26f0e61`.
After v56 authorization, its monitor runs exactly four exclusive,
same-calibration-node, sequential conditions with a 1,200-second hard limit
in the predeclared ABBA order `healthy-no-protection`, `healthy-aegiskv`,
`failed-aegiskv`, and `failed-no-protection`. It deduplicates against both the
live queue and accounting history, and a running condition with at least 180
seconds of stale output only raises `action=no_auto_cancel`. The v57 monitor
contains one `profile_paths` append, requires exactly four paths, and passes
those four paths once to final aggregation; this closes the earlier v41
duplicate-`profile_paths` failure. The monitor and regression test have
SHA-256 values
`38000c06fbd106c97414f069876730e578bbb7e3e02362842a4f18187e166c4e`
and `6d87e248b96b6d013ae5d6781f1fa1728a0d55a761ab05f53a1074ce5ba0d523`.
Neither the v57 gate nor a future four-condition PASS authorizes a performance
claim, paired campaign, formal scale, or paper-ready result.

The global v58 watchdog is independently guarded by
`all-job-watchdog-control-gate-v58.json`, SHA-256
`956e94afd6f3ba806d19f408d4ce4f7d0347c6ae4e714af265f6b0afef451e7a`.
It observes every visible `aegis-*` job and alerts only when a GPU-allocated
`RUNNING` job has at least 180 seconds of stdout staleness. The monitor and
test have SHA-256 values
`3d9ed85a5aeeb6ab7e837554716b9c49474e89e2510c9230bc1d3a15888728bc`
and `54e1b4b6ad9697fe09215d1f37add3b9c6ff158fc94791e9d17391cb81930901`.
The gate requires `read_only_monitor=true` and
`automatic_cancellation_authorized=false`; the script contains no submission,
cancellation, hold, release, or scheduler-update path.

The local v59 `pilot-review-control-gate-v59.json` has SHA-256
`76894dd92fcae49315204daa4d531b5ee0e4b7fb8ab5893386b4ea0bd6525307`.
Only after all four v57 terminal audits and the current-design mechanism gate
PASS may it bind the exact four condition profiles and run an exploratory
engineering review. Its monitor and test have SHA-256 values
`430cfda61786590d842e389df39b6defe0212aa2ff40def1f68f3e8963e2be83`
and `ef108133f2b8fae78b267379112d625407f9c84771f44b82933c24db8dddfea8`.
It is local analysis only, submits no job, and explicitly leaves performance,
formal-scale, and paper-ready claims unauthorized.

The effect-blind v60 `targeted-pilot-gate-control-v60.json` has SHA-256
`0599c3566eda4754796e27442bb6b03aee7875be67df844a717f9114d84d93bd`.
Only after the v59 engineering review and current-design mechanism gate both
PASS may its local planner create one fresh, same-seed `w15/c4` targeted-pilot
plan. The exact ABBA order is healthy no-protection baseline, healthy AegisKV,
failed AegisKV, and failed no-protection baseline; all four conditions use seed
`20260821`, 15 workflows, and concurrency 4. The gate requires
`performance_direction_used_for_selection=false`, submits no job, and cannot
authorize a performance, formal, or paper-ready claim. Its builder, monitor,
and test have SHA-256 values
`ef25a0dbe567ff382315bab345dbf0dd2f9148cfb06b4f376cf2336a257b160d`,
`36b4d07d1cc65d9a466664605e8659adfa51079535f833020575bf45be0ab9ea`,
and `11eca9af3a638b1a5597749cdef951c25a6e1e273850a3512191c69aef877269`.

The v61 `four-condition-series-control-gate-v61.json` has SHA-256
`066a6e5d196e42607f3e4fc610c25e8d0f96e8aa6fce38d7727e612774ce1c99`.
It permits exactly four fresh conditions in this order:
`pilot-no-protection-w15-c4-s20260821-healthy`,
`pilot-aegiskv-w15-c4-s20260821-healthy`,
`pilot-aegiskv-w15-c4-s20260821-failed`, and
`pilot-no-protection-w15-c4-s20260821-failed`. They must run sequentially,
exclusively, on the same calibration node, with at most one in flight and a
1,200-second hard limit. The generated monitor deduplicates exact comments
against both the live queue and accounting history; 180 seconds of stale
output only emits `action=no_auto_cancel`, and no automatic cancellation is
authorized. The monitor generator and test have SHA-256 values
`a20497b9b9d6feb80f82d8d23a3896173233c3a77f1a4a1eeba46e87c828cef9`
and `f61e89c5a71d7e4ba238a2eee6f0d71de439859aa28912c7f4f0e83cb8b426f3`;
the generated r7 monitor has SHA-256
`ef405e4c13ca37ee4aa43e418cf1c6673fd4c67ab8b6c50486e782b4bb5bf966`.

The local v62 `targeted-pilot-review-control-gate-v62.json` has SHA-256
`187a5739d0daa6ef4e501e7d19f2e4cc3dbb95d29cc461b4d1604273aa5eafe7`.
It can run only after all four v61 submissions, terminal audits, mechanism
profiles, and the v61 current-design mechanism gate PASS. The review keeps
positive, null, and negative outcomes, requires
`performance_direction_used_for_inclusion=false`, and submits no Slurm job.
Its only permitted claim class is an honestly labeled preliminary targeted
pilot; general performance, formal-scale, paper-ready, and EuroSys claims stay
unauthorized. The monitor generator and test have SHA-256 values
`1c9d8f75ce179456f5c5c509730c081c4ea61f90865d9998bf81a988868d4cf7`
and `47d6e221d594489fa1c39390e813c573a16703f93dfd19e951b23d80b03609d7`;
the generated r7 review monitor has SHA-256
`9774c7532d476bb8ee4882c3fbeea9a0c5568533c44181ea2d61f024f261d1ff`.

The user units `aegis-a100-v54-physical-profile.service`,
`aegis-a100-v55-mechanism-preparation.service`,
`aegis-a100-v56-mechanism-gate.service`,
`aegis-a100-v57-four-condition.service`,
`aegis-all-jobs-v58-watchdog.service`, and
`aegis-a100-v59-pilot-review.service` are active/running. Their current
evidence is limited to v54 waiting for the v53 transfer-series record, v55
waiting for the v54 profile, v56 waiting for v55 mechanism inputs, v57 waiting
for v56 authorization, the read-only v58 state inventory, and v59 waiting for
the v56/v57 mechanism gate. The expected v53 series, v54 profile/audit, v55
AIMD/input artifacts, v56 authorization, v57 mechanism gate, and v59 review
are all absent.

The user units `aegis-a100-v60-targeted-pilot-gate.service`,
`aegis-a100-v61-targeted-pilot.service`, and
`aegis-a100-v62-targeted-pilot-review.service` are also active/running. Their
current logs contain `WAIT v59_engineering_review=absent`,
`WAIT targeted_pilot_authorization_v60=absent`, and
`WAIT v60_or_v61_targeted_pilot=absent`, respectively. The v60 plan/gate and
authorization, all v61 condition artifacts and mechanism gate, and the v62
review are absent; these services therefore provide static readiness and WAIT
state only, not targeted-pilot runtime or performance evidence.

No current clean-preflight runtime result, config manifest or verification,
recompute calibration, six-run transfer series, schema-v2 physical profile,
AIMD stall calibration, four-condition mechanism gate, fresh targeted-pilot
run/review, replay, or performance artifact exists. A read-only scheduler
check on 2026-08-25 found dependency job 119994 terminal at `COMPLETED/0:0`.
The completed dependency leaves legacy
v19 diagnostic-only clean job 134055 at `PENDING/Resources`, `0:00`, with no
assigned node/GPU; it is not a current r7/v45 submission and was not cancelled.
All ten visible AegisKV jobs remain pending or held with no assigned node/GPU,
so the current r7/v45 monitor still records `WAIT qos_submitted_count=10`.
The active v58 global watchdog therefore sees no GPU-allocated or hung AegisKV
job.
No job was submitted, held, released, or cancelled for this documentation
update. The independent A100 consumer therefore remains fail-closed before
output creation, and none of this source-only/queue-control evidence is
eligible for the V100 consumer.

#### Historical A100-DP2 source and runtime-chain diagnostics

The remaining A100 snapshots in this subsection are retained only as
diagnostics. Any use of “current” inside their recorded snapshot descriptions
means current at that historical snapshot, not current relative to v44-r7.

The earlier lineage snapshot was a schema-only 88/88-pass campaign. A later,
now-historical A100 static snapshot reached 113/113 after creating the
independent execution root
`/home/comp/zhanzhao/aegiskv-execution/qwen3-a100-dp2-20260824-v1-v19-r3`.
At that snapshot, v12 through v19 were applied cumulatively with zero fuzz and
no backup or reject artifacts, and the seven affected source modules passed
143/143 tests. The applicable runtime-chain suites in that already-patched
execution root passed 83/83 targeted tests. Twelve
v17--v19 patch-replay tests are intentionally inapplicable there because the
patches are already applied; their `already applied` failures are not runtime
implementation regressions.
Its exact 108-entry source manifest has SHA-256
`4ea2152206db243fdfbfab5473dbcb753eb8614fa8ba73b9777ecd90a3c40fa7`;
the PASS verification SHA-256 is
`b3c042565b663f51765e5bb48c8b5fb514d6de9d21db9d5a993853782ff3041d`,
and the A100 v19 source-activation SHA-256 is
`b69e2e83bf2c60d60dd2cc62bd65ac942ff9856984714853eeebe0f6c5668890`.
The activation binds the cumulative renderer and strict-audit hashes rather
than a single-patch shadow hash: renderer
`2a802417866d1e4edaa2aa92fb824ed15f353efecbe6cd3b092a5d05e861784b`
and `audit_run.py`
`48538fd53a6e4be4576c74a5d23916506106cfc00dcafa73ca04a7defcd9aba6`.
These hashes and test counts are retained only as historical diagnostics. A
fresh live re-hash of that root against its original v19 manifest now matches
104/108 entries. The four pre-existing drifts are the Mooncake CUDA engine
binary, `aegiskv/experiments/audit_run.py`, and the intranode-NVLink transport
header and implementation. Consequently, v19-r3 is not an active source
lineage and no pending job from it may be consumed. All v19-r3 infrastructure,
preflight, and model-probe evidence below remains diagnostic unless explicitly
superseded by the v20-r4 lineage recorded later in this section.

The A100 source infrastructure now keeps Slurm and host-resource evidence on
the host side of both `model_execution_probe.sh` and
`registered_gpu_copy_probe.sh`: `scontrol`/`squeue`, GPU process snapshots,
co-residency monitoring, and `nvidia-smi dmon` do not run inside enroot. The
container portions perform only CUDA, vLLM, and Mooncake work; registered-copy
also has a static guard against reintroducing those host commands. Registered
copy, recompute, and transfer now require the stable fail-closed contract
`runtime_contract=a100_dp2_source_activated_fail_closed_v1`, rather than a
mutable resource-diagnostic `status`; all three wrong-contract negative tests
reject the input.

A subsequent static audit found and fixed a first-run blocker in that contract.
The runtime entrypoint identities had still named the live producer tree under
`/home/comp/zhanzhao/Mooncake`, although the future Slurm WorkDir and Enroot
mount expose only the isolated execution root. The recompute and transfer
wrappers would therefore have failed closed on path-identity mismatch, and the
live contract itself would not have been visible in the container. All 17
runtime-executed launchers, environments, auditors, and builders are now bound
under the isolated root. Control-only staging, activation, and patch evidence
may remain in the live producer tree. The live producer and isolated activation
contracts are byte-identical with SHA-256
`11892b0b57e263dfc3cf78660084baa9ccff7509fb9c0ec5cb6b26a0df86fbc1`;
the static audit consumes the isolated copy and checks that byte identity.

The A100 model probe distinguishes a requested V2 runner from one observed
during actual initialization with a fail-closed conjunction. The launcher pins
both `VLLM_USE_V2_MODEL_RUNNER=1` and `VLLM_LOGGING_LEVEL=INFO`. After each of
its two workers constructs the LLM, the worker requires
`llm.llm_engine.vllm_config.use_v2_model_runner is True` and records
`model_runner_v2_engine_config_selected=true`. The coordinator must also find
the pinned vLLM GPUWorker INFO marker `Using V2 Model Runner` in that worker's
combined stdout and stderr, and records
`model_runner_v2_log_marker_observed=true`. It derives
`model_runner_v2_observed=true` only when both runtime observations are true;
the request flag is independently required. Unit tests lock both negative
directions: a marker with an unselected engine config and a selected config
without the marker both derive `model_runner_v2_observed=false`. The
config-manifest builder/verifier, registered-copy launcher, and registered-copy
auditor independently require all three runtime-observation fields for both
workers, while a missing marker fails closed. A future A100 SGLSim
config/runtime consumer that reads this probe must impose the same conjunction;
it is not added to or used to relax the current V100-only consumer schema. At
that snapshot, the frozen-source manifest still matched 108/108 because these
campaign-infrastructure and contract fixes were outside its exact selected
entry set; its later 104/108 drift status is recorded above.

The superseded v19-r3 model-probe launcher had an additional fail-closed
runtime contract: it pins `VLLM_ATTENTION_BACKEND=FLASH_ATTN`, moves both the
FlashInfer workspace and `XDG_CACHE_HOME` into the condition-local writable run
directory, and requires each worker result and the coordinator audit to report
`FLASH_ATTN`. This remains historical producer infrastructure; it did not emit
a PASS model-probe artifact and cannot authorize a current-lineage run.

The historical v19-r3 campaign also has a no-GPU, static first-run-readiness
leaf produced by `audit_static_first_run_readiness.py` and covered by
`test_audit_static_first_run_readiness.py`, whose SHA-256 values are
`ce413a2a04caa19265d051e217f9f4e2a3113caef4c19001494ff92f7d4b4fb8`
and `d9271b826387656876e432a2b04d42a0dd06c6c2f1e4660b11bb92ab58172aac`,
respectively. Its final PASS artifact is
`static-first-run-readiness-20260824T081305HKT.json`, SHA-256
`071de31a17d131f6e67d2e0a3847ed305abe0f07ea81d8f8d152dd4f530d93f1`.
All 15 checks pass: they bind the campaign, hardware, design, and runtime
contract; require byte-identical live and isolated activation contracts and all
17 runtime entrypoints under the isolated root; re-hash the 40 activation
references and the exact 108-entry source manifest; validate source
verification/activation, required runtime/model paths, and shell/Python
parsing; import pinned vLLM
`1.2.2+aegiskv.sm89`, PyTorch `2.10.0+cu128`, and CUDA `12.8` metadata with
`CUDA_VISIBLE_DEVICES` hidden and `torch.cuda.is_initialized() == false`; and
confirm that the installed GPUWorker source contains the
`use_v2_model_runner` branch, its `vllm.v1.worker.gpu.model_runner` import, and
the fixed INFO marker. The leaf also verifies that transfer auditing derives
DP count from the hardware profile, that A100 recompute and transfer use their
specialized auditors instead of the DP4 smoke auditor, and that the hashed
Enroot bundle, image, container libraries, and writable-source/read-only-model
mount contract are present.

The A100 recompute and transfer auditors also now dynamically load
`aegiskv/experiments/runtime_profiling.py` from the same isolated frozen source.
They require schema-valid `nvidia-smi dmon` telemetry for both devices, multiple
frames, peak GPU utilization above zero, and peak framebuffer use above zero.
Corrupt telemetry and zero-GPU-activity fixtures both fail closed. These are
auditor gates and static regression tests only; they do not add a telemetry or
physical result to this snapshot, and the frozen 108-entry source is unchanged.

This audit performs zero scheduler queries or mutations, zero GPU queries, and
no CUDA initialization. It performs no model execution, KV movement,
calibration, mechanism run, or performance comparison; both
`submission_authorized` and `runtime_or_physical_result` are false. Readiness
and the campaign manifest reference this leaf in one direction, while its
`activation_submission_fences` check consumes only activation state. The leaf
consumes the isolated activation contract, but neither activation-contract copy
references the leaf, avoiding a circular hash. In particular, this PASS is not
a clean-GPU-preflight PASS and cannot authorize any downstream run.

Before the later resource-state change, five bounded clean-preflight diagnostic
jobs had been submitted to `gpu26`. Jobs 133941 and 133952 failed closed in 9 s and
53 s, respectively, before a physical probe: the isolated Python link was not
relocatable, then `scontrol` was unavailable inside enroot. The link was made
relative and the source was reverified/reactivated, while scheduler/GPU
evidence collection was moved to a host wrapper. Job 133967 then failed closed
in 1 s before CUDA or Mooncake startup because PID 4058790,
`/usr/local/lib/ollama/llama-server`, was already visible on both allocated
A100s, consuming 58,512 MiB and 15,184 MiB, even though the Slurm node-job
snapshot listed only job 133967. Job 134042 repeated that gate more than twenty
minutes later and failed closed in 1 s on the same PID and unchanged memory
occupancy.

After `gpu26` remained continuously Slurm-IDLE and more than one hour had
passed, one low-frequency bounded retry, job 134242, started at
2026-08-24 08:10:52 HKT. It failed closed with exit `1:0` after 1 s and released
the allocation before CUDA, Mooncake, or a model probe started. Its preflight
again found only PID 4058790, `/usr/local/lib/ollama/llama-server`, on the same
two A100 UUIDs with exactly 58,512 MiB and 15,184 MiB in use; the Slurm node-job
snapshot contained only 134242. Its record explicitly sets
`physical_probe_started=false`. At that snapshot this confirmed persistent
unmanaged GPU occupancy outside Slurm; a later external resource-state change
enabled the fresh attempts recorded below.

Those five diagnostic jobs are terminal and their allocations were released.
Their historical five-attempt audit is
`clean-preflight-attempts-audit-20260824T081053HKT.json`, SHA-256
`829c7d05147f30340c49ffe4ff94fc2a7eb7a45d9419ef195726f8b2a2311d1d`;
its 12 internal artifact references re-hash successfully.

Job 134055 is separate from those five attempts. It is a bounded clean
preflight fixed to `gpu25`, pending on `afterany:119994` with zero runtime, no
assigned node, and no GPU allocation. It has no downstream dependency chain
and cannot automatically submit model, config, registered-copy, or calibration
work. Job 119994 remains RUNNING on `gpu25` without GPU GRES. At that older
snapshot, external job 135873 held one A100 on `gpu26`; it later completed
`0:0` at 2026-08-24 19:38:20 HKT and is no longer current. The v19-r3 model
probe 135889 is held before start with no assigned node or GPU. The 081305
static-root activation-contract, readiness, and campaign-manifest SHA-256
values are, respectively,
`11892b0b57e263dfc3cf78660084baa9ccff7509fb9c0ec5cb6b26a0df86fbc1`,
`6f42c7c1d5a95c0de868046134b5bdb3ebd57ee7080700cbe9f4ee1f01cb1b87`,
and `388e79659d78c009df3ccda8a38fe617a544cab573951af7f32bc47feb62db92`.
All 79 expanded recursive artifact references re-hash successfully, including
the manifest, readiness, both activation-contract copies, new leaf, and attempt
roots. The corrected campaign
manifest boundary now states exactly five terminal clean-preflight diagnostics
plus one pending dependency-bound diagnostic and explicitly records that job
134055 holds no node or GPU. This older static root chain does not claim to bind
the later 135852--135889 runtime attempts or their two fix-authorizations.

After the unmanaged gpu26 process disappeared, fresh clean-preflight job 135852
ran for 25 s on both A100 80 GB devices. It physically completed bidirectional
CUDA P2P payload-integrity checks, Mooncake CUDA-engine import, Qwen3-8B
snapshot discovery, Slurm exclusivity, and empty before/after GPU-process
snapshots. Its audit nevertheless remains `FAIL`, SHA-256
`a594f7e1945db04a46a897097090bbd8e9357d0038edd229b1dafaf67dc4e183`:
six of seven gates are true, while `pinned_vllm_v2_runtime=false` because the
old auditor incorrectly searched the relocated Python executable path for the
string `sm89`. Job 135852 is therefore retained only as a failed diagnostic and
must never be relabelled a clean-preflight PASS.

The static authorization
`preflight-auditor-fix-v20.json`, SHA-256
`51dac3c9f72d9bf4ccd3a04fdff20cf22b031490194d637f74d97db9936512d1`,
corrects that predicate to the conjunction of exact vLLM
`1.2.2+aegiskv.sm89`, PyTorch `2.10.0+cu128`, CUDA `12.8`, the exact relocated
Python suffix, and requested V2 runner. The live and isolated auditor/test
copies are byte-identical and independently pass 8/8 tests. This static PASS
only authorized one clean-preflight rerun and does not upgrade 135852 or prove
a runtime/physical result.

That rerun, job 135865, completed `0:0` in 18 s. Its historical
`clean-gpu-preflight-audit.json` has SHA-256
`7aa13432fb99ccdd06086f38ff5865379d9860681a9d13a92d6ccd04c58314ef`
and all seven gates pass, including empty before/after process snapshots. It
was a valid clean-resource, CUDA-topology, runtime-import, and
payload-integrity preflight within the then-current v19-r3 lineage only. Since
that root no longer verifies against its activated manifest, 135865 is retained
as diagnostic evidence and cannot satisfy any v20-r4 prerequisite. It is not
model execution, registered-GPU transfer, calibration, an AegisKV mechanism
result, performance evidence, or a formal/paper-ready result.

Model probe 135871 was correctly bound to the 135865 PASS preflight, but it
remains `FAILED/1:0` after 79 s. Both workers emitted `Using V2 Model Runner`
and began `Loading model from scratch...`; automatic attention-backend
discovery then imported FlashInfer and attempted to create the read-only
`/home/comp/zhanzhao/.cache/flashinfer` workspace. Both workers exited on
`EROFS`, no worker-result artifact was produced, and the aggregate probe is
`FAIL` with SHA-256
`c2a645c1849af6d845ead646911dc82703379c8ec83d2d6c74b0c9a323d09acd`.
Its before/after GPU-process snapshots are empty. These logs diagnose a
post-V2-selection startup failure; they are not a valid model-execution result.

The static authorization `model-probe-runtime-fix-v20.json`, SHA-256
`9c51f387b2faf511ad1e0015e98cae122e3310ecd443f36346d4c672f244d6d2`,
pins `FLASH_ATTN`, relocates FlashInfer/XDG caches to the writable run
directory, and requires the worker result/auditor to confirm that backend. Its
live and isolated entrypoints/tests are byte-identical and pass 18/18 on both
trees. It authorizes only one replacement model probe; it does not upgrade
135871 or authorize registered copy, calibration, a mechanism run, or formal
evaluation.

Model probe 135889 was submitted from the superseded v19-r3 root and was
reversibly held before it started. It remains `PENDING/JobHeldUser`, `0:00`,
with no assigned node or GPU, and must not be consumed. External job 135873
has completed and no longer occupies gpu26. The older gpu25 preflight 134055
remains `PENDING/Dependency`, also at `0:00`, and was not cancelled. None of
these records has produced current-lineage model, config, transfer,
calibration, mechanism, or performance evidence; the current drained gpu26
resource state is recorded below.

The subsequent, now-superseded v20-r4 A100 source lineage used the independent
execution root
`/home/comp/zhanzhao/aegiskv-execution/qwen3-a100-dp2-20260824-v1-v20-r4`.
Its immutable `source-manifest-isolated-v21.json` has SHA-256
`1926f8be994f3b90fca10ae83a889ab858e218fa0d8238dfd49e093ac7c3544e`;
the bound PASS `source-verification-isolated-v21.json` has SHA-256
`ccc27b382691fe3caf05c1e21b7b55f2203d84a9a890150684d66c65b7f08472`.
A fresh live re-hash matches all 108/108 entries, whose canonical digest is
`1119ceb208331bfe6e61729465303bc39e76578b7e72561c6262e7cc08e0ef15`.

The current `aegiskv/experiments/audit_run.py` has SHA-256
`13a83974d169c6cfd46f9f2001ccb467861e67020726f2760f9c8872c7ae99ca`
and passed the producer's 56 focused experiment-tool tests for the
source-owned independent HBM/DRAM semantics. The compatibility evidence class
remains `a100_dp2_v19_source_activation_gate`; the explicit
`activation_revision=v21-current-audit-run` distinguishes the new activation.
The immutable `source-activation-isolated-v22.json` has SHA-256
`993f9631ba6835861ef8c7cc0888f36bdf634e4520a55f4c4cdbefad5e2a98f0`,
status `PASS`, `failures=[]`, and `jobs_submitted=[]`. All ten activated runtime
entries re-hash exactly. The earlier `source-activation-isolated-v21.json`
remains an immutable failed diagnostic and is not upgraded in place.

The initial v20-r4 model attempt used the now-superseded v25 runtime binding.
After correcting the model launcher, the current isolated launcher,
coordinator, and tests have SHA-256 values
`a864c46e868fa3327cd4eaa52ce7e78cb62d26c18894ff8530aa45337dc2b83f`,
`0da0ee57b87537b3dc1bba517ec2730a161b61989d8b60fa0a8cdb3e49855d88`,
and `ef567960f182cb17446777c86ba7bd3decfecf33f1fe94ba8e7375a534f15c00`,
respectively. The current non-superseded A100 campaign suite records 181/181
PASS, plus PASS Python compilation, shell syntax, and diff checks.
These are implementation and static-contract results, not a model PASS or
performance evidence.

The launcher requires exact `AEGIS_A100_SOURCE_ACTIVATION` path and SHA-256
inputs. It checks the compatible
`evidence_class=a100_dp2_v19_source_activation_gate`, exact
`activation_revision=v21-current-audit-run`, `status=PASS`, `failures=[]`,
current source-owned independent HBM/DRAM semantics, and
`jobs_submitted=[]`. It then re-hashes the activation's source-manifest and
source-verification references inside v20-r4, invokes the live
`source_manifest.verify_manifest`, requires exactly 108/108 entries, and
re-hashes all ten activated runtime files.

The current `runtime-binding-isolated-v29.json` has SHA-256
`e399fcbda061d130ea4ba5de18b9249eb197cb2ab148f93164d74489fed4f155`.
It pins exact root-local path, SHA-256, and size for the Python executable,
vLLM package, V2 runner, Torch package, Torch native extension, and Mooncake
engine, while binding the corrected launcher overlay. A current read-only
re-hash matches all six. Earlier v24/v25 bindings remain historical and fail
closed after launcher drift; neither can authorize the corrected retry.

Before worker launch, the coordinator re-hashes all six and must exactly equal
the binding. Both workers independently collect the same six-file identity and
must equal that binding and each other. The launcher also performs the same
fresh 108-entry source-manifest and ten-entry activated-runtime re-hash before
and after model execution. It executes the post-model re-hash even when model
execution fails, and the overall launcher fails unless both model execution
and the post-model lineage check succeed. The model probe still requires an
exact-SHA clean-preflight PASS, two V2 workers, `FLASH_ATTN`, and
condition-local writable caches. A future coordinator result records
`input_lineage.source_activation`, `input_lineage.clean_gpu_preflight`, and
`input_lineage.runtime_binding` as exact path/SHA-256/size identities, with the
same runtime identity at the model top level and in both worker results.

The current one-retry authority is
`model-retry-submission-gate-v30-after-136561.json`, SHA-256
`499dbcc704f42a9bd97aa85b287d694eeec5f42be47f119a86b6104c03ed631e`.
It has `status=PASS`, `failures=[]`, and `jobs_submitted=[]`, and authorizes
exactly one corrected model probe only. Registered-copy, physical-calibration,
mechanism, formal-scale, formal, and paper-ready authorization remain false;
the gate itself is not runtime or physical evidence.

The current-lineage clean preflight, job 135943, completed `0:0` on `gpu26` at
2026-08-24 19:38 HKT after 18 seconds. Its
`clean-gpu-preflight-audit.json` has SHA-256
`25a66049d54b6796b325468709feda4786e48947b3f7dc3bbfe17d6b79070d4d`
and all seven gates pass: two A100 80 GB SM80 devices, bidirectional CUDA P2P
payload integrity, Mooncake CUDA-engine import, pinned vLLM V2 runtime,
Qwen3-8B snapshot availability, exclusive Slurm co-residency, and zero foreign
GPU processes. This is valid clean-resource, topology, runtime-import, and
payload-integrity evidence only. It is not model execution, registered-GPU KV
movement, calibration, a mechanism result, or performance evidence.

The former v26 monitor consumed that PASS and submitted model probe 136561.
The job ran on `gpu26` for 73 seconds, ended `FAILED/1:0`, and released both
GPUs. Its `model-execution-probe.json`, SHA-256
`6fdf1b249fe601c36e86adaadaa08e5fe3578e41823a73f548182f068c20f443`,
records no passing worker result. Both workers failed on the same
`OSError: [Errno 30] Read-only file system` when FlashInfer tried to use
`/home/comp/zhanzhao/.cache/flashinfer`. This is a root-cause diagnostic only:
there is no model PASS and it authorizes no registered copy, calibration,
mechanism, campaign, or performance result.

The corrected launcher now sets `FLASHINFER_WORKSPACE_BASE` to the
condition-local writable `<run>/cache/flashinfer` directory. The v29 binding
and v30 one-retry gate above bind that exact launcher and preserve pre/post
source and activated-runtime re-hashing, including the model-failure path. The
current user queue contains exactly ten pending or held jobs, so the v30
monitor records `WAIT qos_submitted_count=10` and has not submitted the retry.
There is currently no running A100 job.

The post-model boundary continues to use
`audit_model_execution_attempt_v27.py`, SHA-256
`0ca4e59246fb5a2764e2e3fdf51f44b14f3cc650ef32464c8bb96ba9cab98474`,
and its tests, SHA-256
`8121cbfa4bbdf022915ba5f920bd19df954847235c840578f86ef625715a8046`.
The producer records 6/6 targeted tests. The auditor verifies the complete run
checksum inventory; exact top-level model and two-worker runtime identities;
two distinct SM80 A100 workers running Qwen3-8B with all V2 observations; both
pre- and post-model 108-entry source plus activated-runtime verification;
nonzero two-device GPU-utilization and framebuffer telemetry; Slurm
co-residency, allocation, terminal `COMPLETED/0:0`, completion markers; and
both worker logs. Any mismatch fails closed.

Four persistent, fail-closed services now bound the retry chain. v30 may submit
exactly one corrected model retry; v31 only audits its terminal run; v32 may
build a registered-copy gate and submit exactly one copy probe only after a
v31 PASS; v33 only audits that copy and may at most authorize a later matched
physical-calibration gate. They contain no `scancel`, every real job has a
20-minute Slurm hard limit, and no stage authorizes a mechanism run, paired
campaign, formal evaluation, or paper-ready claim. At the current snapshot,
v31 waits for the absent retry submission, v32 waits for the absent retry/audit
lineage, and v33 waits for the absent copy submission. No copy or calibration
job has been submitted.

The post-copy physical-calibration path is now staged as a separate v34--v37
fail-closed chain. `build_physical_calibration_submission_gate_v34.py`,
SHA-256
`85f4fb83e1bbc7443878036b700bfd554e8c74b25830acf09f9082c7c96aeee3`,
accepts only a terminal PASS
`a100_dp2_registered_copy_attempt_audit_v33`. It re-verifies the exact
source activation, clean preflight, model, six-file runtime binding,
registered-copy inner audit, and rebuilt config manifest/verification lineage;
derives `calibration_node` from the audited copy's actual scheduler node; and
authorizes exactly one recompute submission while transfer, mechanism, paired,
formal, and paper-ready authorizations remain false. The corresponding
`a100_recompute_profile.sh`, SHA-256
`1158cefb1cb880e4b72b1644809ade5d536f6c47d7a59e001a8a788d36c3f63b`,
consumes the v34 gate, is pinned to that node, and has a 20-minute Slurm hard
limit.

After recompute, `audit_recompute_attempt_v35.py`, SHA-256
`35a29689b0d0d7cab100b969ee86d6a0bc2230497e841c70cf1f2d8cf2a8f833`,
requires the inner recompute PASS audit, a complete run-directory checksum
inventory, terminal Slurm `COMPLETED/0:0`, the completion marker, and the same
`calibration_node` before it can authorize transfer calibration.
`a100_transfer_calibration.sh`, SHA-256
`f18effa75bdfdc687dba7dc2bf73946f303047a26568d1478a13143435cc4b09`,
consumes that v35 audit and also has a 20-minute Slurm hard limit. The v36
monitor, SHA-256
`cc3f2177ad43394145a300d2d97b7364868680b9df32abe82e1e839f804a01d9`,
runs `aegiskv`, `aegiskv-no-hbm`, and `aegiskv-hbm-reject` serially on that
same node, submitting a treatment only when the QOS has capacity and advancing
only after `audit_transfer_attempt_v37.py`, SHA-256
`d32dd7ba25dcbe426bc4b5e2a767197a81b7526be30d48ac1e5445818575ab9e`,
returns a terminal PASS for the preceding treatment. Only all three v37 PASS
audits can authorize later physical-profile construction; mechanism, paired,
formal, and paper-ready authorization remains false.

The subsequent v38 profile builder is strictly local.
`build_physical_profile_audit_v38.py`, SHA-256
`8a9678f2638b0ae3e00edbdd67c77520482c4dbbf4664cd39c3100b285fb8b1c`,
accepts only an exact three-treatment v37 series-ready PASS. It requires the
three v37 audits to share one nonempty `calibration_node`, the same v35
recompute audit, and its same-node v34 physical-calibration gate before loading
the existing `build_physical_profile_v2.py`, SHA-256
`0b01e5819300c0da548659873572ea42e7204f63c8581f8298f19c8e69700bb8`.
It writes a new A100 schema-v2 profile plus an independent v38 build audit and
refuses partial or pre-existing output. The profile is only a future mechanism
input: v38 leaves mechanism, paired-campaign, formal, and paper-ready
authorization false. `monitor_v38_physical_profile.sh`, SHA-256
`9b35d0b028a7421a198d03d3fa26676f37d1ac16e549be888b112c62803b4535`,
performs no Slurm submission/cancellation or GPU action.

The v34, v35, v36, and v38 user services are active, but their logs only report
missing upstream submission/recompute/series-ready records. No v34 gate or
recompute submission, v35 attempt audit, v37 series-ready artifact, or v38
profile/audit exists, so these services have submitted no job and performed no
profile build. This chain contains no
`scancel` and consumes the current source-activation lineage rather than the
superseded live `activation-contract.json`. These are static producer and
control-path contracts, not a schema-v2 profile, four-condition gate, runtime
measurement, or performance result.

The A100-local v39 contract now adds a pre-treatment AIMD local-stall
calibration builder, `build_aimd_stall_calibration_v39.py`, SHA-256
`a71cc961f09eb578485d7b8aae9ed33f087fb61ee3cbae92d1fd22e497041e44`.
It can create the immutable `aimd-local-stall-calibration-v39.json` only after
the same-node v38 schema-v2 physical profile and its independent PASS audit
exist and agree. It scans the physical profile's hash-bound gate logs for
successful `target_local=true` HBM `allocation` or `reclaim` operations,
requires at least two valid samples drawn from at least two distinct physical
gate logs, and sets `AEGIS_AIMD_LOCAL_STALL_MS` to their duration-millisecond
median. The output inherits and hash-binds the v38 source manifest, config
manifest, config verification, physical profile, and physical-profile audit;
it also records `builder_and_verifier` as the exact isolated campaign-local
builder path, SHA-256, and size. The subsequent v39 preparation gate
deterministically rebuilds and requires the supplied config verification to be
the exact PASS result, independently re-hashes `builder_and_verifier`, and
requires it to equal that exact campaign-local builder. An outer calibration
object whose PASS fields were forged after builder drift therefore fails
closed. The calibration records
`performance_direction_used_for_selection=false`, so no treatment direction
or pilot outcome can select the threshold. The calibration-builder test has
SHA-256
`dd79214ef615ac048dd767faf86feaaff94e7b54aa0d94e04d86c9bbd9f1b6b1`.

The pinned config supplies a five-second AIMD interval and a stable-window
threshold of three. The builder therefore freezes a 25-second post-workload
observation tail, `5 s * (3 + 2)`. A future formal-mode pilot condition keeps
services and GPU telemetry alive through that tail and writes
`mechanism-observation-window.json`; v41 requires its observed duration to be
at least 25,000 ms. Mechanism profiling may include AIMD/control/co-residency
events through the observation end, while workload/performance metrics remain
bounded by the original summary start/finish window. The tail is therefore
not added to reported workload performance time.

No v38 profile or audit currently exists, and an explicit artifact scan found
no `aimd-local-stall-calibration-v39.json`. This is a fail-closed producer
contract, not an observed threshold, physical profile, mechanism result, or
performance result.

The v39 layer prepares, but does not authorize, a future A100-DP2 mechanism
run. Audit found that the frozen formal renderer still requires four DP
workers, so invoking it directly for system-DP2 would fail before a valid
four-condition run. The frozen 108-entry source and the v30--v38 chain were not
modified. Instead, the live and isolated campaign overlays add five
byte-identical, fail-closed A100 wrappers:
`render_a100_single_node.py` at SHA-256
`31868d6a1252504428f180179edcf0f504eaa2a7d10e4ac42a10642d386afb89`,
`render_a100_run_config.py` at
`0f14183a50f83a3e9d12e9f2570084327fe1b44d98557f5db242a14b6a439aee`,
`run_a100_workload.py` at
`16cecfd12ad591f0b37e7ba5e17cbe724fb843eb23e48c039816ee87c2e3005a`,
`audit_a100_formal_run.py` at
`419853a607696a35a726ac8e71906c1681abd2ca3913d72798e073330f012d4d`,
and `materialize_a100_full_stack_v39.py` at
`f42e9767c231a23aab4e1eef6380bd61e0153165e8161cff34cc226b487a7300`.

The renderer and workload wrappers replace only exact, uniquely matched
DP4-to-DP2 cardinality checks; any frozen-source text drift fails closed. The
deployment wrapper requires exactly two workers and writes
`allow_degraded_single_survivor=true`, so a future failed DP2 treatment can
explicitly run with its single surviving worker. The A100 strict auditor first
runs the complete frozen auditor and may repair only its exact residual
store-pool-count/admission-policy check and its public-vLLM-version check after
revalidating system-DP2 and local `vLLM 1.2.2+aegiskv.sm89`; every other frozen
strict check is retained. The materializer refuses to generate an A100 launcher
unless frozen `aegiskv/slurm/full_stack_smoke.sh` still has SHA-256
`a337af1133698eebc12a12175b1f218e94b6ff96d4edce2acf3f027f3eea2063`.
The updated A100-local workload extension also performs one exact replacement
that retains the complete frozen `campaign_condition` object in each formal
summary, in addition to its condition ID. Its regression test has SHA-256
`1b3db72b9d82587794c5bc0d2e3411ef70cc940772638aa16b6ec0beda702176`.
This closes the generic pilot reviewer's requirement that
`summary.campaign_condition` equal the frozen-plan condition, without modifying
the frozen general workload runner. Future v39/v40 artifacts must hash this
updated extension before any runtime can use it.

`build_mechanism_preparation_gate_v39.py`, SHA-256
`11e2f5ed5ff3ac9eb34507ccb8f261977a35b49ae37b7576b52264d31323c487`,
can authorize local workload materialization only after the v38 schema-v2
profile and independent v38 audit agree; the profile's source/config lineage
re-hashes; config verification deterministically rebuilds to the supplied PASS;
all 108 frozen source entries freshly verify; the model/vLLM identity matches;
and the fixed Weka trace, training workload, predictor, slack calibration, and
training report match SHA-256 values `29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276`,
`570f7a1b14e225dbbca23a92177fa59c7789bde7d51468adf04c63c766f4283c`,
`184fe5a0c093dcc3f0ddb83ba0dcc53af3d9f5bd9a6faeb2d39ccb4e5c282b88`,
`ac6c679f2d2cb5f1969323ca65a26ac47fd50b62b50c06ffd0afbdd82e64bb76`,
and `c195afd63b4eed65bd3a72f99cfb3ae8473b4b347cbd2000bdeda925d26253bf`,
respectively. It also requires and re-hashes the v39 AIMD calibration, checks
that its physical/config lineage matches v38 exactly, requires a positive
threshold and the 25-second observation contract, and rejects any calibration
selected using a performance direction. Its own test has SHA-256
`34bc32c9557b8ba635bbf6025dc372966e483d27453c0f37ce086fe972ca2529`;
the two affected modules pass 5/5 in both live and isolated trees. Its output
keeps
`mechanism_submission_authorized=false`, all downstream/formal authorizations
false, `jobs_submitted=[]`, and `runtime_or_physical_result=false`.
`monitor_v39_mechanism_preparation.sh`, SHA-256
`6d9c898cde2178394502fc89a5863a70601ee06e27ba9c8d164730a57f741da4`,
contains no `sbatch` or `scancel`; it can only create immutable local Weka
workload inputs after that preparation gate. Before the preparation gate it
locally builds the missing AIMD artifact only when the v38 prerequisites are
present; pre-existing output is refused.

The pre-sync v39--v42 systemd user-service processes were intentionally stopped
to load this calibrated-threshold/25-second-observation code. Their unit
instances show `inactive/dead` after the controlled SIGTERM at 21:36:22 HKT;
they are not the effective current monitors. At 21:36:23 HKT four replacements
started as independent PPID-1-owned processes: v39 PID 2658593, v40 PID
2658594, v41 PID 2658595, and v42 PID 2658596. A live process check observes
all four in `Ss` state. At the final 21:48 HKT read-only refresh, their logs
respectively report `WAIT v38_physical_profile_missing`,
`WAIT mechanism_inputs_ready_v39_missing`, `WAIT v40_plan_or_gate_missing`,
and `WAIT v40_or_v41_gate_missing`. This controlled replacement submitted,
cancelled, and released no Slurm job. No v39 calibration, preparation-gate,
materialized-workload, readiness, GPU-action, or mechanism-gate artifact
exists. This is static preparation/readiness only, not physical, mechanism,
performance, formal, or paper-ready evidence.

The subsequent v40/v41 layer is a fail-closed, A100-local four-condition
runtime chain; it is not a completed mechanism experiment or a performance
campaign. Ten new campaign-overlay files (five runtime/control files and their
five tests) are byte-identical between the live campaign and v20-r4. The
runtime/control identities are `a100_four_condition_run_v40.sh` at SHA-256
`e26cd83134d6aa7f297ab037161dd8ef4adf69c0ab5929cb89a763da6dde1ce9`,
`build_mechanism_submission_gate_v40.py` at
`096ef99d29892623a873016a0d6ab5d55aef346e910d96e40a9a37a88b37dd45`,
`audit_four_condition_attempt_v41.py` at
`fc1db3ee5781e89c4b9e1ffdb00bd82afe74ebc04050d9f24f427a561e5a806c`,
`monitor_v40_mechanism_gate.sh` at
`258b65df247b8b5df9cda389006e27e29f4eccc40a4c958631613f18a351659c`,
and `monitor_v41_four_condition_series.sh` at
`c2673163a8f6fcf2318a359b8c6543500df5f80160a7b951b27b819ba8df25e1`.
Its fail-close regression test has SHA-256
`ed204723de00e9a71608559109deb50e08ffaa2825ef448194f545b4d1e33e65`.
The updated v40 builder test has SHA-256
`5a8c0ea989398c4a59c595ba801bda12fab168b521c41493d7af1edbe406d0a4`.

v40 can freeze a plan and submission gate only after the absent v39
mechanism-input readiness artifact passes and re-hashes its v38 schema-v2
profile, source/config lineage, workload, predictor, slack calibration, and
training report. It additionally requires the exact v39 AIMD calibration
reference shared by the preparation gate and readiness artifact, re-hashes it,
and freezes that reference into both the plan and submission gate. The
immutable plan now explicitly sets `phase=pilot`, stores
the exact physical `calibration_node` as its top-level `node`, and stores the
causal failure request and delay at the top level for the frozen strict-audit
contract. It fixes four workflows at concurrency four, seed 20260821, and this
exact counterbalanced order: healthy no-protection, healthy AegisKV, failed
AegisKV, and failed no-protection. v41 binds the routing seed to the same
20260821 value. Both failed conditions target
`weka-94055207ca7ba1e6-s4-b1` with a 5000 ms
`sigkill_process_group` delay. The v40 gate requires the same physical
calibration node, at most one in-flight condition, and a 1200-second hard limit.
It authorizes only these four bounded mechanism submissions; paired-campaign,
formal-scale, performance-claim, formal, and paper-ready authorization remain
false. It separately authorizes a local engineering review only after a PASS
mechanism gate, while `pilot_performance_claim_authorized=false`. The v40
builder and monitor perform no Slurm or GPU action.

Before considering a new submission, v41 first searches the user's queue for a
pre-existing job with the exact condition- and gate-derived Slurm comment. It
records and resumes that exact job even when the submitted-job count is already
ten; only a genuinely new `sbatch` is placed behind the below-ten QOS fence.
This ordering prevents both a crash-window duplicate and a dead wait after a
successful submission whose local record was not yet written. v41 otherwise
submits the conditions sequentially. For each condition it requires terminal
Slurm `COMPLETED/0:0`, positive elapsed time no greater than 1200 seconds, the exact
calibration node, strict-audit PASS, a checksum covering the complete run
directory, and empty pre/post GPU-process inventories. It stops on any invalid
terminal attempt and contains no `scancel`. Only after all four terminal audits
pass may it build four condition profiles and invoke the existing
four-condition mechanism auditor; even that output keeps the performance claim
disabled. Every per-condition submission exports the exact
`AEGIS_AIMD_LOCAL_STALL_MS` value from the hash-bound v39 calibration. The
launcher rejects profile/config/calibration drift before runtime; the terminal
audit requires the same calibration reference in the runtime-input artifact
and the complete 25-second observation-window artifact before allowing profile
construction.

The effective PPID-1 v40 and v41 replacement processes currently report only
`WAIT mechanism_inputs_ready_v39_missing` and `WAIT v40_plan_or_gate_missing`,
respectively. The old systemd unit instances remain inactive as described
above. There is no v40 plan or gate,
v41 submission or terminal audit, condition profile, or four-condition report,
and no job was submitted by this layer. The five focused test modules pass
12/12, and the 181/181 applicable suite plus Python compilation and shell
syntax checks pass. These results establish static/runtime-chain readiness
only, not A100 runtime, physical-profile, mechanism, or performance evidence.

The local-only v42 review is a second postcondition over the same four
predeclared v41 runs; it does not submit or rerun them. Only after the existing
current-design four-condition mechanism gate is PASS,
`monitor_v42_pilot_review.sh`, SHA-256
`54ee293fdae46902186417a36711918fa973ac0841df058c7ba5e84734c997a4`,
requires all four v41 submission records, terminal PASS audits, and mechanism
profiles. It requires the four observed profile path/SHA-256 pairs to equal the
mechanism gate's exact four `condition_profiles` references before creating a
local submission manifest. It also re-hashes the v40-bound
`review_pilot.py` and `analyze_campaign.py`, whose current isolated SHA-256
values are `524ccb5f8df0d2cccc8c584bbc9a10790c4788b2eb8b7a6236fecb5be7b2cf40`
and `16e788c28092f1029b6a794a039b2a22d70e569760c71cfc95a373ff40a5800c`.

The reviewer requires `phase=pilot`, the exact four treatment/failure pairs,
four distinct job IDs, strict-audit PASS, and each formal summary's complete
`campaign_condition` to equal its frozen plan condition. If all checks pass, it
may report at most one exploratory AegisKV difference-in-differences comparison
plus healthy-path and resource overhead from those same runs. The output remains
engineering validation only: it does not authorize a performance claim,
formal-scale submission, formal result, or paper readiness. The v42 monitor and
its test, SHA-256
`372c2001042b8f1c3c781409b768a8ad50226890de5ef5f7b9a3d4fc4e22caf5`,
contain no `sbatch` or `scancel`.

The effective v42 monitor reports only `WAIT v40_or_v41_gate_missing`. No v42
submission manifest, analysis directory, or engineering-review artifact
exists. The current AIMD-propagation set consists of 22 named producer/test
files across v39--v41 (including the unchanged v40 monitor and its test); live
and isolated copies compare byte-for-byte 22/22. The full applicable live and
isolated suites each pass 181/181; Python compilation, shell syntax, and
repository diff checks also pass. These remain static/runtime-chain results,
not a pilot result or a performance result.

At the historical 2026-08-24 21:01 HKT resource snapshot, independently
rechecked one minute later, Slurm reported `gpu26` as `IDLE+DRAIN`, with no
allocated TRES and reason
`Kill task failed [root@2026-08-24T20:51:41]`. The reason timestamp
matches the terminal time of another user's job 136732,
`mediq-actorhf-npcvllm-smoke-b2-fix`, which completed `0:0` after 24:37. The
producer later observed a transient `MIXED` state with one A100 assigned to
another user's job. The final read-only refresh at 21:48 HKT instead reports
`gpu26` `IDLE`, empty `AllocTRES`, and no job on the node. These transitions are
external scheduler/resource state, not AegisKV activity. Direct SSH/process
inspection remains unavailable because of cluster authentication. No process
was killed and no node drain/resume state was changed during this
synchronization.

The v30 corrected model retry remains QOS-blocked at ten submitted jobs and is
not pinned to `gpu26`; its `sbatch` has no node list. A later physical
calibration gate derives and fixes `calibration_node` from the scheduler node
of the validated registered-copy attempt. The changing `gpu26` record is
therefore scheduler/resource diagnostics only and is not AegisKV runtime,
physical, mechanism, or performance evidence.

The earlier gpu25 exclusion and
`clean-preflight-scheduler-amendment-v23-job-135943.json` remain historical
scheduler-protection provenance. Job 135943 ultimately executed cleanly on
`gpu26`; the amendment is not itself runtime, mechanism, or performance
evidence.

The staged A100 four-condition profile builder has also been corrected before
any mechanism run. The live and v20-r4 copies of
`build_four_condition_mechanism_profile.py` are byte-identical at SHA-256
`9345cfd238cdeb679ea7ab721e9886de94a0c111669b68618b79c99cc75020f9`;
their byte-identical tests have SHA-256
`477d02ea3a52bbc8c805f33b3ff3ecc62b7a7ea0171c64bc56153d1fea3694b5`.
The builder now records the authoritative wire enum `BYPASS=0`,
`ADMIT_DRAM=1`, `INVALID=2`, and `ADMIT_HBM=3`. An HBM-tier decision may only
be 0 or 3, and a DRAM-tier decision may only be 0 or 1; `INVALID` and cross-tier
values fail closed instead of being counted as admissions. The profile now
also binds the v39 local-stall calibration and the post-workload observation
window, requires the deployed AegisKV threshold to equal the physical median
exactly, and keeps only AIMD/control/co-residency events—not ordinary workload
performance records—from the 25-second tail. The producer records 15/15
targeted profile tests.

The matching A100 four-condition gate independently enforces the same mapping.
The live and v20-r4 copies of `audit_four_condition_mechanism_gate.py` are
byte-identical at SHA-256
`af0fae8cf38225b909f43494b408daad9cfd7aa39ee18c6f34c877b32432baa6`;
their byte-identical tests have SHA-256
`5d37ded2d6d1f0b9b8552d36c3d6c78373152ff624d41bb4c9266c7903557b49`.
Every condition profile must carry the exact authoritative mapping. The legacy
`0=ADMIT_HBM` mapping, any nonzero `INVALID`, or any nonzero
`UNEXPECTED_TIER_DECISION` fails closed. The stable gate check name is
`runtime_admission_decision_wire_values_verified` and must not be renamed.
The additional stable hard check `stable_windows_additively_increase_alpha`
requires the Aegis profiles within the four-condition set to aggregate
`alpha_increase_count > 0`. Together with the existing pressure-window check
for multiplicative alpha decrease and quota contraction, this requires both
AIMD directions in any future physical profile. None of the prior ten stable
mechanism-gate check names was removed or renamed; this is an additional
producer-side requirement and is not added to the current V100 consumer. The
new A100-only hard check
`physically_calibrated_local_stall_threshold_applied` additionally requires
every one of the four condition profiles to carry a positive calibration,
prove it was deployed, and prove its runtime windows match. It is additive:
the prior ten stable check names, including their exact spellings, remain
unchanged, and neither this check nor the prior additive-increase check changes
the V100 schema or consumer. The producer records 32/32 PASS per tree across
the current builder and gate tests (15 builder and 17 gate tests).

The same overlay now emits a per-pool physical AIMD timeline containing the
observation time, pressure and reasons, alpha, quota and allocation, CUDA
free/total memory, stable-window count, and pause/reclaim/resume evidence.
AegisKV conditions fail closed when the timeline is missing or any window is
malformed. The A100 gate also checks that the calibrated local-stall threshold
was applied; this does not alter the authoritative
`source_owned_independent_hbm_dram_v2` semantics or the wire values
`BYPASS=0`, `ADMIT_DRAM=1`, `INVALID=2`, `ADMIT_HBM=3`. These fields are staged
producer infrastructure, not an observed AIMD timeline or mechanism result.

The live `activation-contract.json`, SHA-256
`11892b0b57e263dfc3cf78660084baa9ccff7509fb9c0ec5cb6b26a0df86fbc1`,
still binds the superseded v19-r3 static paths and hashes. Consequently, two
full-suite readiness tests currently fail closed as intended; twelve v17--v19
patch-staging tests are inapplicable because those patches are already applied.
Job 135943 is a v20-r4 clean preflight and its batch script does not consume the
mechanism-profile builder, so the old live contract cannot authorize this
runtime. The staged v34--v37 calibration path likewise does not consume this
old contract: it rebuilds current source/model/runtime/config lineage from the
terminal v33 copy audit. Before any A100 mechanism campaign, the current
builder, gate, and their test hashes must still be freshly frozen and bound
into the new campaign lineage.

The historical V100 v11 Slurm chain is now a terminal old-lineage diagnostic,
not an in-progress calibration campaign. Job 131875 ran for two seconds and
ended `FAILED/1:0`; its only stdout diagnosis is an intra-engine SHA mismatch
(observed `bd7bc...`, expected `e2f42...`). Job 131876 is
`DependencyNeverSatisfied`, jobs 131877--131878 remain behind unmet
dependencies, and job 131879 is held. Those latter four jobs have zero runtime
and no assigned GPU. None was cancelled, released, or otherwise modified by
this documentation sync, but this chain cannot produce current-design
source/config/profile/gate evidence. The V100-only consumer therefore remains
fail-closed until r2, or a later explicitly superseding current-source
TP1/PP1/system-DP4 lineage, passes the complete physical profile and mechanism
contract. Source activation and a clean-submission gate alone are insufficient.
A read-only resource check on 2026-08-25 also
resolves the apparent gpu24 GRES mismatch: although this Slurm version prints
`N/A` for job 137123 (`cvp_y2`) in `squeue %b`, both its job TRES and gpu24
`AllocTRES` record `gres/gpu:v100s=4`. The other user's job started at
2026-08-24 22:56:42 HKT and has nominal end 2026-09-07 22:56:42 HKT, so it
owns all four V100S devices on gpu24. On gpu23, job 120072 owns one GPU and the
node records `gres/gpu:v100s=1`, with nominal end 2026-09-04 11:43:52 HKT.
Consequently no four-V100S node is currently free. These are real external
allocations, not leaked, dead, or GPU-owning AegisKV jobs. This documentation
sync performed no Slurm action. The A100 attempts above likewise do not modify
the V100 queue, its consumer, or its profile/check-name contract.

The final recorded v20-r4 boundary consisted of source identity, static
activation, a six-file runtime binding, one physical clean-resource/topology
PASS, a failed model diagnostic, and waiting fail-closed producer stages. The
v19-r3 clean preflight 135865, failed model startup 135871,
held-before-start model job 135889, and all v20-r4 monitor/queue observations
are historical diagnostics and cannot enter the v44-r7 lineage. They produced
no importable physical profile, four-condition mechanism result, replay, or
paper-ready evidence. The ten stable mechanism-gate check names were not
changed. The SGLSim V100 consumer remained V100S-TP1/PP1/system-DP4-only and
failed closed on every A100-DP2 profile.

### Recorded L40S-DP2 v23-r1 source/queue lineage (not consumed)

The original L40S job 135847 was submitted for live-root v22 recompute
calibration, but that manifest is no longer valid. A fresh read-only re-hash
matches only 107/108 entries: it expected
`aegiskv/experiments/offline_placement_oracle.py` SHA-256
`a486fc9cbb986db74bc18618ef2ee03d1b3362c414ca07bb460b0a5a2f7aa1cf`,
while the current exact oracle is
`2cc6233594bf7d4eb6c5e78b27683c7993d54405a0dc7f019fc0f143b6d4285f`.
Job 135847 was held before allocation at zero runtime, so the stale recompute
stage consumed no GPU and is not calibration or physical evidence.

The current L40S source identity is the physically separate execution root
`/home/comp/zhanzhao/aegiskv-execution/qwen3-l40s-dp2-20260824-v1-v23-r1`.
All 108 frozen manifest entries have inodes distinct from the live source. Its
`source-manifest-isolated-v23.json` has SHA-256
`0cc8f2a063728f89cef61c88c0ef0e7681790d47818a94d2a24df39ab1ab2f91`
and entries digest
`5a7a797e2e8f1ee04379741349f59cfe8919d761b525f8fc070c3af76f5c85e0`.
The 108/108 PASS `source-verification-isolated-v23.json` has SHA-256
`accf91bfde1c2f7d475d151e8bd5ad0e74bace14528165fd1ecff7eef670e48a`,
and the frozen entries remain unchanged. The former
`source-binding-isolated-v23.json` and model-submission v25 snapshot are
superseded before submission: the model-runtime Python entry point was an
absolute symlink into the live checkout, while v25 did not content-hash the
runtime that a future job would execute. Neither artifact is current
submission authority, and no v25 job was submitted.

The current `source-binding-isolated-v27.json` has SHA-256
`b00dfb56ec08bbcc158f9050877d2deb1c24ec0c5543f8d20383be9068bf9dba`.
It binds the fresh 108/108 PASS
`source-verification-pre-model-submission-v27.json`, SHA-256
`3f94848a7b58294945fc4eac329fe98387fc1bb69299059269065fd7999b90e4`.
The model-runtime entry point `.venv-vllm-sm89/bin/python` is now a relative
execution-root-local symlink to
`../../artifacts/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12`.
The binding records exact path, SHA-256, and size for six runtime artifacts:
the Python executable, vLLM package, V2 model runner, Torch package, Torch
native extension, and Mooncake engine. The coordinator and both workers must
re-hash all six and match the binding. The runtime artifacts may still share
physical inodes with the live checkout because a complete runtime copy failed
closed on disk quota; v27 therefore provides execution-root path and content
identity, not a claim of physically independent runtime storage. This binding
authorizes only one fresh two-worker L40S model probe and does not authorize
registered copy, calibration, a mechanism gate, or a paired campaign; no
calibration or result is shared with A100 or V100.

The L40S model gate is a campaign overlay outside the frozen 108 entries. Its
launcher, coordinator, and tests have SHA-256 values
`8488e4d16a6f53e941553aa451bb310275ec3a49c61b968b214df5ea4bf49abe`,
`a3e0b36bc9ea71046d2a0ed38d1594f905e97319f1636deaef7bc019633db67d`,
and `2bcfef63b787034276b477861c6a68a6d7006b6f3317d8313bcc41346db3e8ce`,
respectively. Live and isolated copies are byte-identical; the producer records
14/14 targeted model-gate tests in each tree. The isolated source verification
remains 108/108 PASS. Ruff was unavailable in that environment.

Before model or CUDA work, that launcher hash-checks the supplied source
binding, requires its exact v23-r1 execution root, hardware/campaign/current
design identity and model-probe authorization, re-hashes the bound manifest,
and freshly requires all 108 source entries to verify. After model execution,
it repeats exact source-manifest verification and again requires 108/108 PASS.
A future coordinator and both workers must also reproduce the exact six-entry
`runtime_identity` from the v27 binding. A future model output must include
exact path/SHA-256/size records at `input_lineage.source_binding` and
`input_lineage.source_manifest`. These gates do not constitute a model PASS.

The self-contained submission-time snapshot
`submit_isolated_v27_model.sh` has SHA-256
`6d5e5bc477326adfad223ff1718c15d05e276366fea1593130601dae38b7d71b`.
It pins the launcher, coordinator, v27 binding, frozen manifest, fresh source
verification, and topology-preflight hashes and requests exactly one node, two
L40S GPUs, 32 CPUs, 160 GiB memory, and 20 minutes. It contains no scheduler
mutation.

The current static `model-submission-gate-v27.json` has SHA-256
`2ecf7ea01a70a6d8b53bed476afd58551ddb348a9665dbaa47b1d256a657a9f1`;
its corrected independent
`model-submission-gate-v27-verification-v2.json` has SHA-256
`20f43ee5c8a5ccb5fe0f68aa3c5bf92d69f9b5645d30b9f5e2114b6877a74b5c`.
The corrected verification has 62/62 true checks, `failures=[]`, and
`jobs_submitted=[]`; the earlier uncorrected v27 verification is not a current
root. The gate authorized only one v27 model probe in the v23-r1 root.
`v27-model-submission.json`, SHA-256
`4d2ffd0dc0a6a1b61b7d54b18f5199afa46f1d59bbed74fd9201d862b8664d4d`,
records that bounded submission as job 136567. Registered copy, physical
calibration, the mechanism gate, and the paired campaign remain unauthorized;
`formal_ready=false` and `paper_ready=false`. Submission is queue evidence,
not model runtime or physical evidence.

The post-model attempt auditor `audit_model_execution_attempt_v26.py` is staged
in both the live campaign and isolated v23-r1 root, with byte-identical SHA-256
`fb8a5fc86bcbc6be4f1ce321a64a0ae50ba48c1ef1d99e3eaefaa08426614dbd`.
Its byte-identical test file has SHA-256
`31217fd45e31f8110f702ec436e00466cf5af6d3cb309e7543e7d16fdd672706`.
The producer records 6/6 focused post-model-auditor tests and 120/120 PASS for
the complete live L40S static campaign, plus PASS Python compilation,
byte-comparison, and diff checks.

The live and isolated L40S mechanism-gate copies are byte-identical at SHA-256
`8f73254a4b0d0493be84d91f543ea6611755441b15c04a0f578af26991a4dc37`;
their byte-identical tests have SHA-256
`459f4c55dac9b0bc0228ab5cd691835e8ee911eaa45302e3868d235825bfbf7c`.
They likewise add `stable_windows_additively_increase_alpha` and require the
Aegis profiles in the four-condition set to aggregate a positive
`alpha_increase_count`. The 120/120 total includes its zero-increase
fail-closed regression. This is staged static gate evidence, not a mechanism,
runtime, physical, or performance result.

For the current v27 model attempt, v26 fail-closes on the supplied immutable v27
submission gate and corrected verification, including nested
`submission.jobs_submitted=[]`, and exact source-binding, source-manifest,
topology, and six-artifact runtime identity. After the model run, the auditor
re-hashes all six bound runtime files and requires both the model top-level and
the two worker `runtime_identity` objects to equal the v27 source binding. Its
worker-runtime-identity drift regression fails closed. It also requires
`formal_ready=false`/`paper_ready=false`, two distinct passing L40S vLLM V2
workers, 108-entry source verification both before and after the model run,
empty before/after GPU-process inventories, exclusive Slurm co-residency,
complete run-directory checksums, nonzero telemetry from both GPUs, clean
worker logs, terminal `COMPLETED/0:0` Slurm evidence, and all completion
markers. Only a v26 PASS may authorize exactly one registered-copy probe;
physical calibration, mechanism-gate, paired-campaign, formal, and paper-ready
authorizations remain false. No v26 model-attempt PASS artifact exists yet, so
this is staged audit infrastructure rather than model/runtime evidence.

Job 136567 is currently `PENDING/Resources`, `0:00`, with no assigned node or
GPU. The active fail-closed `aegis-l40-v28-attempt-monitor.service` runs
`monitor_v28_model_attempt.sh`, SHA-256
`55c660b12da795ee4bc877fdfaced10e3f7a8b26f3208702864b986de35388df`.
It waits for the job's terminal state and only invokes the v26 model-attempt
auditor on a successful completion; a failed terminal state authorizes no
registered copy. The Slurm job has a 20-minute hard limit. This monitor does
not submit registered-copy, calibration, mechanism, or campaign work.

The staged registered-copy overlay is now bound exclusively to that v26 audit.
Its launcher `registered_gpu_copy_probe.sh`, auditor
`audit_registered_gpu_copy.py`, and test file
`test_audit_registered_gpu_copy.py` have SHA-256 values
`5067912c890df7ff4db31619e6e9576a264b8778563e3b0b196afd4d85d38c77`,
`cf3d5cc68b1d0a7c0a4e29bfad488d0e380d847f08709580a23280897048dcc4`,
and `21d7ffc3c979697c1b9574b875c23b5ec2e954a97bf1ec85038bd1218e7b7034`,
respectively. All three live/isolated pairs are byte-identical. The producer
records 15/15 PASS across the registered-copy and v26 targeted tests.

Both the launcher and auditor require the exact v26 evidence class
`l40s_dp2_v23_model_execution_attempt_audit` and reject the legacy
`l40s_dp2_model_probe_attempts_audit`; the latter has an explicit negative
regression. They require current source-owned independent HBM/DRAM design
semantics, exact source-binding and model-result hashes,
`registered_copy_submission_authorized=true`, every calibration/mechanism/paired
authorization false, and `formal_ready=false`/`paper_ready=false`. Because no
v26 PASS or model result exists, this registered-copy chain remains staged and
unauthorized in practice; it is not registered-copy, calibration, mechanism,
performance, formal, or paper-ready evidence.

After the site QOS rejected an additional submission without creating a job,
an attempt was made to reuse 135847 by adding an exact-JobId dispatcher to the
live launcher. That attempt is not effective: Slurm runs the batch-script
snapshot captured at submission, so a later filesystem edit cannot repurpose
the pending job. The dispatcher was removed, and 135847 was re-held before
start. The live recompute launcher was restored byte-for-byte to SHA-256
`e4e1e92cd6606a4d12a020bd0965470bbc6a69e62a6d5397c1e456dff6271f44`.
No job was cancelled or newly created, and no GPU was consumed.

The failed attempt is recorded in `pending-job-135847-repurpose-v24.json`,
whose current SHA-256 is
`96fd5a9331157e06aeba3c3fd2979a78f95f892670ef34befd8a00592ea79e9e`,
evidence class `l40s_dp2_pending_job_repurpose_attempt_audit`, and status
`FAIL`. Its four explicit failures are
`qos_max_submitted_jobs_prevented_a_new_model_job`,
`slurm_uses_the_batch_script_snapshot_captured_at_submission`,
`pending_command_update_is_not_supported`, and
`filesystem_dispatcher_cannot_repurpose_job_135847`; it records
`model_probe_submission_authorized=false`. Job 135847 is currently named
`aegis-l40s-held-v22`, with comment
`held-stale-v22-source-drift`; it remains `PENDING/JobHeldUser`, `0:00`, with
no node or GPU. The independent v27 model probe is now submitted as job 136567
from the isolated v23-r1 root, but it remains pending without runtime or GPU
allocation.

This L40S lineage remains source plus queued model-probe infrastructure. Job
136567 is submitted but still `PENDING/Resources`, `0:00`, with no node or GPU;
the v28 monitor is waiting to fail-close audit its eventual terminal state.
There is no L40S model PASS, registered-copy result, current config,
calibration, four-condition gate, runtime/physical/performance result, or
formal/paper-ready evidence. The separate A100 clean preflight 135943 has
completed, while its first model attempt 136561 is a failed diagnostic and its
corrected retry remains unsubmitted behind the ten-job QOS cap. Neither lineage
has a model PASS, registered-copy result, calibration, mechanism, or
performance result.
The isolated manifest remains SHA-256
`0cc8f2a063728f89cef61c88c0ef0e7681790d47818a94d2a24df39ab1ab2f91`,
with 108/108 PASS and entries digest
`5a7a797e2e8f1ee04379741349f59cfe8919d761b525f8fc070c3af76f5c85e0`.
The current SGLSim consumer remains
V100S-TP1/PP1/system-DP4-only, rejects L40S inputs fail-closed, and retains the
same schema and ten stable mechanism-gate check names.

### Isolated A100-DP2 preflight consumer

SGLSim now has a separate, validation-only A100 entry point:
`preflight_mooncake_a100_dp2.py`. It does not import `run.py` or the V100
profile class, and it accepts only `A100-1node-TP1-PP1-systemDP2`, two SM80
A100 devices, Qwen3-8B FP16, `source_owned_independent_hbm_dram_v2`, and wire
values `BYPASS=0`, `ADMIT_DRAM=1`, `INVALID=2`, `ADMIT_HBM=3`.

Before creating an output directory, it re-hashes and independently validates
the source and config identities, v46 runtime binding and model-submission
gate, v47 model-attempt audit and V2 worker identities,
registered-GPU-copy evidence, v50--v54 same-node calibration chain,
schema-v2 coefficients, v55 physical AIMD median, four condition-local strict
audits, and the A100 mechanism gate. Superseded v20-r4/v43 binding or retry-gate
classes fail closed. Its exact dmon
arguments are:

```text
--id=0,1 --select=pucmet --delay=2 --options=DT --format=csv --filename=gpu-telemetry.csv
```

The model-attempt audit must additionally expose a PASS
`details.per_gpu_telemetry` result with no failures, exact observed GPU IDs
`["0", "1"]`, at least two complete frames, and positive per-device
`peak_gpu_utilization_pct` and `peak_fb_memory_mb`. Producer failures such as
`per_gpu_telemetry` or `per_gpu_runtime_activity` are therefore consumable
fail-close evidence, never an aggregate success. The consumer then reads the
model, registered-copy, and condition CSV files itself and reproduces the
model audit's exact per-device peaks and frame count. Both GPU 0 and GPU 1 must
independently have multiple complete frames, positive peak SM utilization, and
positive peak framebuffer use. A missing producer field, producer/raw drift,
legacy whitespace output, `pucvmet`, or an aggregate-only nonzero summary fails
closed.

The A100 physical-profile contract is affine per path rather than accepting a
single `effective_bytes_per_ms`. It requires exactly two independent physical
repeats of each transfer treatment (`aegiskv`, `aegiskv-no-hbm`, and
`aegiskv-hbm-reject`): six wrapper audits, six generic audits, and six v53
terminal attempt audits in total. Each repeat remains condition-local and
binds the same source manifest, config manifest, PASS config verification,
v46 runtime binding, v46 model-submission gate, hardware profile, and
recompute-profile lineage. The frozen generic transfer wire retains the field
name `model_retry_gate`, but the referenced artifact must be that independently
validated v46 gate; this spelling is not permission to consume an old retry
gate. The v53 series represents each treatment as a two-element attempt list;
the v54 audit binds the same six attempts as one flat, duplicate-free set and
joins them through the v52 compatibility view to the original v51 recompute
attempt and v50 physical-calibration gate.

Every repeat has one decimal Slurm job/run ID and exactly one derived stream
ID, `slurm-job:<run-id>`. The run ID must occur in the run-directory identity,
all physical transfer events in that run must use that same stream ID, and
`calibration_transfer_ordinal` must be contiguous from zero in actual issue
order independently for each event. A chunk, request, worker, or worker-log ID
cannot create another stream. Reusing a run directory, run ID, stream ID,
wrapper, generic audit, or gate log fails closed.

Affine fitting uses the two `aegiskv` repeats for both Peer-HBM paths and the
two `aegiskv-no-hbm` repeats for both direct-DRAM paths. The two
`aegiskv-hbm-reject` repeats remain independently required physical fallback
evidence, but cannot be counted as extra DRAM streams in the fitted path. Thus
each declared path model is reproduced from exactly two real transport
streams. Each of Peer-HBM protection, direct-DRAM protection, Peer-HBM
recovery, and DRAM recovery must independently provide:

- `per_transfer_setup_ms`;
- an additional `first_transfer_warmup_ms` for transfer ordinal zero; and
- `steady_state_bytes_per_ms`.

The declared duration formula is `setup + first-transfer warm-up +
bytes/steady-rate`. SGLSim re-reads the hash-bound, run-local physical gate
logs and fits the later samples using exactly two run-derived streams, four or
more steady-state samples, and two distinct payload sizes (`R² >= 0.80`), then
derives the ordinal-zero median residual as warm-up. The declared treatment,
repeat count, exact stream-ID list, model, all three metrics, and raw fit must
agree. A legacy single-bandwidth profile, a one-run treatment, a fabricated
chunk/worker stream, a duplicate stream, a missing setup/warm-up field, or
folding either overhead into the steady-state rate fails closed.
Protection models remain healthy-path costs only, while only the two recovery
models enter Algorithm 2 recovery timing.

For current producer inputs, the A100-only validator also requires a complete
deployment `hbm_quota_aimd` configuration. `total_hbm_bytes` must equal the
pinned per-target HBM capacity; the interval, memory threshold, alpha bounds,
decrease factor, increase step, and stable-window threshold must equal the
config manifest; and `local_stall_ms` must equal the v55 physical median. Every
Aegis condition must provide the PASS strict-audit block at
`measurements.aegiskv_data_plane.aimd`. SGLSim requires exactly the two Router
pools, joins their IDs and timelines to the independently hashed condition
profile, and replays every emitted pressure reason, alpha transition, stable
window, quota, and aggregate. A no-protection condition may omit this Aegis
data-plane block, as current Mooncake does, or emit only an explicit disabled
all-zero block. The raw local allocation/reclaim duration samples are not
duplicated in the emitted strict summary; they remain covered by the hashed
PASS strict audit and are never synthesized by SGLSim.

A successful invocation writes only an immutable
`a100_dp2_preflight_<lineage-sha16>` directory containing a normalized profile
and preflight manifest. The manifest explicitly leaves trace loading, model
fitting, replay, figure generation, and performance claims unauthorized. The
25-second post-workload AIMD observation tail must be present but cannot be
included in the performance summary window.

The current source authority is the statically ready v44-r31 source/control
lineage. Its clean-only gate and queued-zero-runtime job 139550 are not an
SGLSim consumer preflight, physical profile, or performance artifact. The r30
resource-contaminated clean failure, r28 zero-runtime model job 139406, r26
PASS model audit, and r27 pending diagnostic are separate lineages and cannot
supply r31 with model, registered-copy, or calibration evidence. Historical
r7/r12/r13 and other intermediate lineages remain separate.

No single current lineage supplies the required config manifest/verification,
model and registered-copy evidence, recompute calibration, six-run transfer
series, affine physical profile/audit, physical AIMD calibration,
four-condition mechanism gate/profiles/current strict audits, fresh SGLSim
preflight, or targeted-pilot result. A future consumable artifact set must be
built wholly from r31 (or a later explicitly superseding lineage), contain two
independent repeats per treatment, retain the run-derived stream and actual
issue-order fields above, include all four affine models, and provide both
condition-local strict-audit population and AIMD blocks. No RTX4090 TCP value
or cross-lineage A100 diagnostic is an eligible default or fallback. The
formal A100 entry therefore exits before creating output. This is intentional;
no A100 replay or performance result exists.

The existing `run.py` path remains a V100S-TP1/PP1/system-DP4-only consumer.
Its schema, output naming, runtime contract, and ten stable check names are
unchanged by this independent A100 preflight.

#### Immutable calibrated-suite binding and static runner fence

`calibrated_suite_contract.py` and `bind_calibrated_suite.py` provide the next
static-only layer for future larger experiment matrices. The binder accepts
exactly one of two independent inputs: a reproducible PASS
`sglsim_a100_dp2_consumer_preflight_v2`, or the existing V100 schema-v2 import
after rerunning `preflight_v100_config`. Declaring one consumer kind while
supplying the other lineage is rejected. The A100 path replays the complete
producer validation from the original nine artifacts rather than trusting the
outer preflight PASS; the V100 path retains its existing V100-only importer,
schema, hardware label, and ten stable mechanism-check names.

The immutable matrix names the exact model and revision, the calibrated
system-DP, planned maximum in-flight request counts, planned per-workflow
failure-probability values, HBM/DRAM capacity fractions, and deterministic
matrix/failure-schedule seeds. Model or system-DP extrapolation beyond the selected physical lineage
fails closed; each derived capacity records both its fraction and calibrated
byte value. The binding identity covers the consumer manifest/payload, physical
profile SHA, complete calibration identity, and normalized matrix. Both the
binding directory and future runner output name are hash-derived and existing
paths are never overwritten. Each planned cell ID additionally includes the
suite-binding identity, so identical axis values from A100 and V100 cannot
alias one another.

`preflight_suite_runner` re-hashes the binding and all original consumer
artifacts, reconstructs every matrix cell as `PLANNED_NOT_RUN`, and returns an
in-memory static preflight only. It exposes no timing, throughput, WCT, recovery
benefit, or other performance field; trace loading, model fitting, replay,
result output, figures, and claims all remain false. Missing physical inputs
therefore prevent even suite materialization, while the current r31 clean
submission gate cannot masquerade as the independent SGLSim consumer
preflight. No replay or synthetic performance artifact was created while
testing this layer.

The preceding single-lineage snapshot was independently revalidated with
`uv run --extra test pytest -q test/unit/srt/test_aegis_*.py`: 556/556 tests
passed; the four focused contract files passed 134/134, the A100-specific
contract tests passed 58/58, and both
`uv run python ... --help` entry-point checks passed. The unchanged V100-only
entry point `run.py` has SHA-256
`2e48034ae44b3830b5007be903db81d83e0365ab5496e392c21c044e950239c3`, and
its importer has SHA-256
`48bd25196abe4ded3bd6d22add479ba54372e98d6c2113001db8ed00b91c95db`.
This was static validation only: no replay, figure, performance run, or Slurm
action occurred, and every calibrated-suite cell remains `PLANNED_NOT_RUN`.

#### Multi-lineage large-model/config readiness campaign

`scale_campaign_contract.py` and `bind_scale_campaign.py` add a separate
composition layer above the immutable single-lineage binder. Its declarative
requirement manifest does not contain a built-in model list: every entry must
state its consumer kind, hardware label, exact model name/scale/revision/dtype,
TP/PP/system-DP/worker layout, physical-profile schema/semantics/SHA,
calibration and mechanism-gate identities, lineage identity, workload axes,
and one calibrated-suite artifact reference. Each referenced suite is
independently rebuilt with `validate_suite_binding` and statically preflighted
with `preflight_suite_runner`; an outer `PASS` or matching filename is never
trusted.

The campaign rejects duplicate requirement IDs, duplicate physical lineages,
reused suite bindings, mismatched model revisions or layouts, cross-hardware
assignment, duplicated local/global cells, and declared or hard cardinality
overflow. A missing suite reference or artifact produces a machine-readable
`BLOCKED_MISSING_LINEAGES` readiness result with no cells and creates no output
directory. Once every declared lineage is present, the immutable campaign
binding still contains only globally unique `PLANNED_NOT_RUN` cells; every cell
explicitly carries its physical lineage and suite-binding identity. Trace
loading, model fitting, replay, result output, performance/formal claims, and
paper readiness remain false in both the binding and runner preflight.

This layer is large-model/config **readiness infrastructure**, not permission
to extrapolate one profile across models or system-DP layouts. A larger model,
new revision, TP/PP/DP topology, or future hardware target must first produce
its own valid physical consumer input and single-lineage suite binding. No
replay, figure, performance result, physical KV movement, or Slurm action was
performed while adding this contract.

Static validation for this generic layer now passes 19/19 focused
multi-lineage tests. The current combined and repository-wide counts, including
the paper declaration layer below, are recorded after its evidence boundary.
These are implementation checks only: every materialized campaign cell remains
`PLANNED_NOT_RUN`, and no trace, replay, figure, performance output, or Slurm
job was created.

#### EuroSys paper-matrix requirement and readiness artifact

`eurosys_paper_requirements.json` applies the generic multi-lineage schema to
the current `aegiskv/paper/sections/05-evaluation.tex`; no model list is built
into the validator. The three declarative requirements are independent:

- Qwen3-8B FP16 on one 2xA100 node, TP=1/PP=1/system-DP=2 with two Workers;
- Qwen3-8B FP16 on one 4xV100S node, TP=1/PP=1/system-DP=4 with four Workers;
- Qwen3-32B BF16, TP=1/PP=4/system-DP=4 with 16 Scheduler processes and a
  32,768-token experiment cap.

Every requirement carries its own hardware/model/layout, model revision and
`config.json` SHA slot, physical-profile schema/semantics/SHA, calibration and
mechanism-gate identities, lineage identity, workload axes, and suite-binding
reference. Unknown measured values are JSON `null`; the generic contract emits
field-specific readiness gaps instead of accepting a sentinel SHA or inferred
coefficient. A calibrated-suite descriptor now includes the independently
validated model-config SHA, so even the same model name and revision cannot
silently cross a different frozen config. No A100, V100S, or future 32B
binding can be reused for another requirement.

The shared paper axes exactly retain token projection scales 0.004, 0.008,
0.016, and 0.032; workload shapes `(15,4)`, `(30,4)`, `(60,1)`, `(60,4)`, and
`(60,8)`; and the Pilot/Central/Scale/Ablations table of 4/70/60/50 paired runs.
The A100 and V100S 8B entries retain the paper's 40,960-token executable cap;
the 32B point uses 32,768. `eurosys_paper_readiness.py` parses those declarations
from the paper source, delegates every physical check to
`scale_campaign_contract.py`, and produces only a no-cells readiness audit.
`audit_eurosys_paper_requirements.py` materializes that immutable audit and
returns a nonzero status while the campaign is blocked; it never invokes the
suite runner, trace loader, model fitter, replay, or result writer. The audit
also hash-binds the paper validator, generic scale contract, and calibrated
suite contract used to derive it, so validator drift is not accepted through
an old outer status.

The checked source has SHA-256
`232dd00ff775801166fb40f2e7a17617cc654efe0273c2d44dbdabc4bc68f7d8`.
It now explicitly declares the separate A100 engineering-pilot lineage, its
TP1/PP1/system-DP2 layout, and its non-formal evidence boundary, so A100 has no
paper-source alignment gap. The 32B model/layout/context declaration also
matches, including 16 Scheduler processes, but its concrete hardware remains
unspecified in the paper and is intentionally a machine-readable paper-source
gap. The requirement manifest has SHA-256
`3d740ce84f94212c785c0d48db963726ec2d06c66efeda6b3c7f8e7ce345765a`;
the reproducible `eurosys-paper-readiness.json` has SHA-256
`f9cceb4cd9993dad4da9f3ab4271840778c117ae20dfd406a42dfe304760037d`.

The current readiness status is `BLOCKED_PAPER_REQUIREMENTS`: all three
requirement IDs are missing and none is bound. The immutable audit retains
exactly 33 machine-readable readiness gaps. Each 8B requirement still lacks
the exact model revision/config SHA, physical-profile SHA, calibration and
mechanism-gate identities, lineage identity, numerical failure/capacity/seed
axes, and calibrated-suite reference. The 32B requirement additionally lacks
its hardware label and profile schema class. Consequently the artifact has
`planned_cell_count=0`, `cells=[]`, no runnable or partially runnable cells,
and false trace/replay/result/performance/formal/paper authorization.

The current r31 producer state does not close the A100 gaps. Clean job 139550
is only queued at zero runtime; there is no r31 clean/model/config/
registered-copy PASS, recompute, six-transfer calibration, schema-v2 profile, or
four-condition gate. r30 and r28 remain non-joinable diagnostics. The current
V100S v20-r2 producer likewise does not close the V100 requirement: source
activation and the clean-preflight-only gate are PASS, but its singleton
monitor is QOS-blocked at 10/10 with no V100 job or submission record. There is
no observed clean/model/config/KV-transfer, recompute or four-path calibration,
schema-v2 physical profile, or four-condition mechanism gate. Legacy V100 job
131875 terminated FAILED after
two seconds; 131876 is `DependencyNeverSatisfied`, 131877--131878 have unmet
dependencies, and 131879 is held. None of this old-chain evidence is
consumable. This work performed no replay, figure, performance
run, physical KV movement, or Slurm action. A complete future requirement may
expand the shared paper axes into globally unique `PLANNED_NOT_RUN` cells, but
the full paper campaign remains fail-closed if any one lineage is absent.

Static validation passed 27/27 focused scale/paper tests, 131/131 combined
A100/V100/scale/paper contract tests, and 583/583 repository
`test/unit/srt/test_aegis_*.py` tests. Both scale and paper CLI `--help` checks,
Python compilation, and readiness-artifact reconstruction pass. The unchanged
V100-only `run.py` and importer retain SHA-256
`2e48034ae44b3830b5007be903db81d83e0365ab5496e392c21c044e950239c3`
and
`48bd25196abe4ded3bd6d22add479ba54372e98d6c2113001db8ed00b91c95db`,
respectively; their schema and ten stable mechanism check names were not
modified.

#### Static Mooncake output-accounting bridge

`python/sglsim/srt/simulation/aegis_mooncake_bridge.py` is an A100-only,
in-memory validator/helper layered after the independent
`sglsim_a100_dp2_consumer_preflight_v2`; it is not an alternate V100 importer
or a replay entry point. Its input class is `MooncakeA100BridgeInputs`, with
exact paths for the SGLSim preflight manifest, Mooncake schema-v2 physical
profile, PASS four-condition mechanism gate, failed-AegisKV condition profile,
and an optional component profile. The condition profile must refer to a PASS,
hash-bound `strict_audit_report`. That strict audit is accepted only as the
condition-local runtime population source, never as preflight evidence.

The bridge maps Mooncake physical-profile names as follows:

| Mooncake profile path | SGLSim accounting identity |
|---|---|
| `peer_hbm_protection` | `source_to_peer_hbm`, HBM, `source_d2d`, `hbm_physical_copy` |
| `direct_dram_protection` | `source_to_store_dram`, DRAM, `source_d2h`, `dram_physical_put` |
| `peer_hbm_recovery` | `peer_hbm` recovery hits from `hbm_physical_load` |
| `dram_recovery` | `dram` recovery hits from `dram_physical_load` |

The physical profile must contain all four
`affine_setup_first_transfer_warmup_v1` models. For each path, the bridge
independently checks `per_transfer_setup_ms`,
`first_transfer_warmup_ms`, and `steady_state_bytes_per_ms` against both the
top-level metrics and runtime-consumption mapping. The failed condition's
strict audit must contain
`measurements.aegiskv_data_plane.protection_effectiveness` with
`evidence_scope=same_strict_run_gate_events`, exact
`source_to_peer_hbm`/`source_to_store_dram` rows, and an exact aggregate row.
Every row contains successful protection copy count/bytes, recovery hit
count/bytes, whether amplification is defined, and copies/bytes per hit. The
bridge recomputes all ratios and aggregates, verifies the recovery tier's
`planned_recovery_event_count`, `planned_recovery_request_count`, and
`planned_loaded_bytes`, and then requires the mechanism profile's four
`source_owned_paths` counters to match the same strict-run population. An old
strict audit without this block, a zero-filled undefined ratio, or any
protection/hit population crossing fails closed.

The bridge lineage ID is SHA-256 over canonical JSON containing schema,
campaign, hardware, design, the independent preflight file SHA and reproducible
preflight lineage ID, source-manifest SHA, config-manifest SHA, PASS
config-verification SHA, physical-profile SHA, mechanism-gate SHA, failed
condition-profile SHA, its strict-run-audit SHA, and the optional component
profile SHA. The gate and preflight must hash the same four condition profiles;
the failed condition must independently rehash the same source/config/profile/
AIMD lineage and one of the four preflight-bound strict audits.

Optional component input uses evidence class
`mooncake_a100_transfer_component_profile_v1`. An available
`hbm_physical_copy` path must provide complete affine profiles for exactly four
components: target allocation; source pack, defined as
`source_pack_allocation_ms + source_pack_duration_ms`; engine transfer; and
target finish. Their setup, first-transfer warm-up, and reciprocal steady-rate
terms must add back to the aggregate HBM profile. The current monolithic
`dram_physical_put` duration is explicitly recorded as unavailable unless a
future producer supplies and hash-binds a complete valid breakdown. Missing
component values remain `None`/listed in
`remaining_profile_dependent_fields`; they are never synthesized as zero.

`build_aegis_output_contract_from_mooncake` only constructs the existing
metadata accounting dictionary in memory. It still requires separately
supplied, direction-blind paired whole-workload WCT and failure-local effects,
and leaves replay, figures, performance claims, formal readiness, and paper
readiness false. This matches the current accounting semantics: physically
observed recovery hits require the same artifact's successful-protection
population, a zero-hit path emits `amplification_defined=false` with both
per-hit values `None`, and every retained pair contains both whole-workload WCT
and failure-local measurements. SGLSim is deliberately more conservative than
Mooncake's formal paired analyzer: because this is metadata-only descriptive
accounting rather than confidence-interval inference, both
`general_advantage_supported` and `performance_claim_authorized` remain false
even when both point estimates improve. No replay/runtime execution path was
changed for this synchronization.

Focused tests are in
`test/unit/srt/test_aegis_mooncake_bridge.py`; they cover the valid mapping,
strict-audit-as-preflight substitution, old audits without the population
block, condition-population crossing, source/config/profile/gate hash drift,
partial component profiles, explicit DRAM unavailability, and missing paired
effects. The A100-DP2 suite additionally rejects missing frozen AIMD fields,
missing Aegis strict blocks, DP4/cross-topology pool counts, altered controller
law, and nonzero no-protection AIMD. Together with
`test_aegis_output_contract.py` and the V100-import cross-hardware rejection,
the current static check passes 68/68. No RTX4090 diagnostic is an input to
this bridge.

After the importer succeeds, the formal replay must receive both emitted
artifacts. The runner re-hashes the config, import manifest, physical profile,
source manifest, source verification, pilot/runtime artifacts, and context
audit before loading the Weka trace:

```bash
uv run python AegisKV/experiments/real_weka_full_design/run.py \
  --config /path/to/new-v100-sglsim-config.json \
  --import-manifest /path/to/new-v100-sglsim-import-manifest.json
```

Its output basename is
`full_393_workflows_v100_<physical-profile-sha12>`. The directory must not
already exist; an explicit `--output-dir` is accepted only with that exact
hash-derived basename.

For a development replay:

```bash
uv run python AegisKV/experiments/real_weka_full_design/run.py \
  --config /path/to/new-v100-sglsim-config.json \
  --import-manifest /path/to/new-v100-sglsim-import-manifest.json \
  --limit-workflows 20 --skip-dataset-sha \
  --output-dir /tmp/smoke_20_workflows_v100_<physical-profile-sha12>
```

## Evidence boundary

The dataset workflow topology and within-workflow timing are real, but the
corpus does not provide
a cross-workflow arrival clock, so independent workflow arrivals are an explicit
deterministic Poisson projection. The primary token counts are scaled by the
fixed 0.004 projection and mapped to the physical Qwen3-8B FP16 KV identity.
Peer-HBM protection, direct source-to-Store DRAM protection, physical
Peer-HBM recovery, Store DRAM recovery, and recompute coefficients are bound to
distinct, hash-verified artifacts from one coherent V100S campaign. The primary
simulation preserves the physical TP1/PP1/system-DP4 layout; any cross-layout
sensitivity is a separate fail-closed profile. Consequently, this is a
calibrated metadata-level mechanism and optimization study, not a paper-ready
physical end-to-end recovery result. The output manifest explicitly records
whether AIMD used the diagnostic request-count proxy or the hash-bound physical
headroom/stall model.
