# Independent Mooncake RTX4090 PP2 lineage

This is an isolated, fail-closed import contract for one preliminary targeted
physical campaign.  It does not replace, extend, or relax the V100-only
consumer, and it does not mix V100S or A100 artifacts into the lineage.

## Identity and evidence boundary

- Hardware label: `RTX4090-2node-PP2-systemDP2`.
- Model/layout: Qwen3-8B, two nodes with two RTX4090 GPUs each, TP=1, PP=2,
  system-DP=2. Stage 0 is on `gpu11`, stage 1 on `gpu13`; each node hosts two
  pipeline replicas.
- Design semantics: `source_owned_independent_hbm_dram_v2`.
- Frozen source-entry digest:
  `9ccd7da9e870c5ee962c235c50e2cfc5ef0763422f3c7f1389c838174c58e0e3`.
- Profile semantics: `rtx4090_pp2_registered_hbm_tcp_four_path_v1`.
- Evidence role: preliminary targeted physical diagnostic. It is neither
  formal nor paper-ready and does not authorize SGLSim replay or figures.

The historical source manifests are re-hashed and their 175 entries must have
the exact canonical digest above. The current mutable Mooncake checkout is not
required to equal that historical freeze, because producer hardening after the
run legitimately changed live files.

## Pinned producer inputs

| Gate/artifact | Job | SHA-256 |
|---|---:|---|
| topology gate | 135388 | `de04291819c704f98223dbd386c01597ddabd47436bb39a0f32899f0043891cd` |
| Qwen3-8B PP serve | 135393 | `d17bf9b8b82ab63e720400d3c60e148897a95919b5947892057fd0804e78a76f` |
| recompute calibration | 135399 | `5d90ecbc98cfdd7850882bd91ccaef45181bc8c2ce8ac47e763d3d2678b4b9ec` |
| Peer-HBM protection calibration | 135408 | `5fbfa14612858feca93ecf23b840153c80c1def7c6e4e8a48f1fefce085db706` |
| direct Store-DRAM calibration | 135416 | `06461b479c1753fbf9bdb139afabc33fb903537dc94d712ab66059c218e3fb2b` |
| independent HBM plus direct DRAM | 135430 | `1e20edbb89dba7781fa19442bb0f0278beb0a5df69957e43bad47d4441fe4a83` |
| AIMD HBM_ONLY direct reclaim | 135440 | `bceb2c28648dd67d919b84afd2e7fea448e478dceaf2b5014be6bd74cee1bcd0` |
| whole-pipeline HBM recovery | 135452 | `d92210155a6704a77725f3c5d35f985cb34dae21ae586470e90a7bef8dbe2d08` |
| long-context series audit | — | `7a1afb53aa2935cee0aa6d7f8756c0fea354661caa8727566cf6e9de68423d22` |
| human preliminary report | — | `658a58e855841c61794e50d57b90350d764c32ba91e1e390e37c90ca523d6033` |

The importer recursively re-hashes artifact records nested below the gates,
including all 48 Peer-HBM sample results and the physical recovery logs. The
two `full-request-v1` control reports are separately pinned as
`5380571d...45ec8` and `b2249f48...8c64c4` and cannot be pooled with the
long-context series.

## Transport truth and future-schema normalization

The 135408 source and target are distinct RTX4090 device-memory endpoints; the
target is a dynamic registered tensor. Nevertheless, all 48 result artifacts
record `transfer_protocol=tcp`, and the 135452 recovery agent/events use the
same transport. A separate read-only four-second Slurm diagnostic, job 135738,
completed `0:0` on `gpu11` and `gpu13` with four RTX4090 GPUs. On both nodes,
`nvidia-smi topo -p2p p` reported GPU0↔GPU1 as `NS`, and PyTorch reported both
peer-access directions as false. That job had no standalone immutable producer
JSON; the profile therefore marks this as a bounded report-attested topology
diagnostic rather than upgrading it to a formal calibration artifact.

The old frozen 135408 top-level gate is not required to contain fields added
later. The importer derives the facts from all 48 physical samples, then emits
the stricter normalized contract:

- `transfer_protocol = "tcp"`
- `physical_path = "registered_gpu_memory_over_mooncake_tcp"`
- `direct_cuda_p2p_ready = false`
- `exact_d2d_p2p_contract_ready = false`
- `simulator_rate_default_eligible = false`
- bandwidth semantics are two independent PP-stage
  registered-HBM-over-TCP lanes.

Thus “physical Peer-HBM” here means registered HBM endpoints with actual bytes
moved and recovered through Mooncake TCP. It must not be relabelled CUDA
P2P/D2D or GDR, reused for V100 P2P, or treated as default/paper-ready SGLSim
calibration.

## Four independent rates

The profile preserves four rates and their roles:

| Rate | Derived value (bytes/ms) | Permitted use |
|---|---:|---|
| Peer-HBM protection | 2,751,026.7145 | healthy-path copy pressure/observation |
| direct DRAM protection | 49,557.1935 | healthy-path copy pressure/observation |
| Peer-HBM recovery | 3,393,023.4226 | Peer-HBM `tier_recovery_ms` only |
| DRAM recovery | 109,288.7600 | DRAM `tier_recovery_ms` only |

Protection rates are not subtracted from Algorithm 2 Gain and are never used
as recovery rates. Recovery rates do not cross tiers. Recompute is independently
fit at 0.1053353979 ms/token with 14.4731129171 ms nonnegative fixed overhead.

## Preliminary result and mandatory control

The predeclared `long-context-recovery-critical-v1` series used 2,200 words
(2,933 actual prompt tokens), `max_tokens=192`, and a 2,000 ms failure delay.
All four counterbalanced seeds were retained and positive; each AegisKV pair
physically recovered 2,048 cached tokens after dual-stage SIGKILL, while its
baseline recovered zero.

- Mean failure-added latency saved: 139.814 ms (6.765%).
- 95% t interval: 52.446–227.182 ms (2.852–10.678%).
- Exact bootstrap 95% interval: 93.189–186.439 ms (4.652–8.878%).
- Mean healthy-path overhead: 23.876 ms.

This four-pair targeted result remains preliminary. The separate decode-heavy
`full-request-v1` controls saved -43.687 ms and +34.281 ms, for a -4.703 ms
mean. They show a near-neutral/unstable effect and must remain visible and
separate rather than being hidden or aggregated with the positive series.

Only `figure-v3` is accepted:

- PNG `32444be339779ffdf09fb44beaca24360813a046ec594e779de88505a049549c`
- PDF `c290af80a6cfe9e99a87a305af5fb211364cf9b3cd4e321f5647c1d1fb3e1345`
- CSV `d66b2dbeda7fa286f8072473153c03670a05a2f82ba636217112d3bceb930182`
- audit `edc50da1ab0bb7edf323155318085ddb78851def6b8e5f4b075c0cfc40480751`
- renderer `b7326ac961aa02014d1b7eceef724d26e234f99564a00e9ec6f25347a8ca2daf`

The audit must state `invalid_attempt_count=0` and “No structurally invalid
attempt supplied performance values.” Older `figure/` and `figure-v2/` are
diagnostic only and fail closed.

## Import and validation

The currently validated SGLSim-side bundle is
[`imports/rtx4090_pp2_ab6666ff0e7f1085/`](imports/rtx4090_pp2_ab6666ff0e7f1085/):
its `profile.json` SHA-256 is
`ab6666ff0e7f1085edfc90da82e7f7ca053c75beba9b9db82da097479f542387`,
and its `import-manifest.json` SHA-256 is
`96f525ae51164de234c615d2ef95e84b9145c8026f2446006010a5ac33cba549`.
The directory suffix is derived from the profile hash and a second import to
the same output root is rejected rather than overwritten.

Import into a new SHA-derived, non-overwritable directory:

```bash
uv run python \
  AegisKV/experiments/real_weka_full_design/import_mooncake_rtx4090_pp2.py \
  --mooncake-root /home/comp/zhanzhao/Mooncake \
  --output-root /path/to/rtx4090-imports
```

Validate the output without constructing a runtime object:

```bash
uv run python \
  AegisKV/experiments/real_weka_full_design/consume_mooncake_rtx4090_pp2.py \
  --profile /path/to/rtx4090_pp2_<profile-sha>/profile.json \
  --manifest /path/to/rtx4090_pp2_<profile-sha>/import-manifest.json
```

The consumer reconstructs the entire profile from the pinned source artifacts
and compares it byte-for-semantics with the imported profile. By default it
does not materialize timings and never runs replay. The optional explicit
acknowledgement `registered_hbm_over_tcp_diagnostic_v1` permits only construction
of an `AegisFullDesignConfig` diagnostic object; it still records
`replay_authorized=false` and `replay_performed=false`. There is intentionally
no replay or figure function in either 4090 entry point.
