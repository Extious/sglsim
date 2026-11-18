# Strict-Final Real-System Recovery Matrix

Collect real SGLang Host recovery data-plane evidence that can upgrade metadata-only SGLSim policy rows, without changing the metadata-only SGLSim runtime.

This plan intentionally keeps SGLSim metadata-only rows separate from real
SGLang recovery-path evidence. Passing this matrix supplies real-system
Host restore evidence; it must not be represented as SGLSim physical tensor
copy rows unless the final audit is explicitly updated to consume that
evidence.

## Submit Commands

### strict_final_same_write_best_effort_r1_host_invisible_lookup_disabled

```bash
ssh hkbugpumanode 'cd ~/aegiskv_sglang_small && RUN_LABEL=strict_final_same_write_best_effort_r1_host_invisible_lookup_disabled PORT=32820 REMOTE_BACKUP_PORT=33820 ENABLE_REMOTE_BACKUP=1 DISABLE_FAILOVER_REMOTE_LOOKUP=1 HICACHE_STORAGE_PREFETCH_POLICY=best_effort REMOTE_BACKUP_REQUEST_AWARE=1 REMOTE_BACKUP_BUFFER_GB=16 NUM_REQUESTS=4 MAX_TOKENS=512 PROMPT_REPEAT=256 UNIQUE_REQUEST_PREFIX=0 FAILURE_DELAY_S=12 READY_TIMEOUT_S=2400 REQUEST_TIMEOUT_S=900 sbatch slurm/aegiskv_dp2_failure_smoke.sbatch'
```

### strict_final_same_write_best_effort_r1_host_visible_lookup_enabled

```bash
ssh hkbugpumanode 'cd ~/aegiskv_sglang_small && RUN_LABEL=strict_final_same_write_best_effort_r1_host_visible_lookup_enabled PORT=32821 REMOTE_BACKUP_PORT=33821 ENABLE_REMOTE_BACKUP=1 DISABLE_FAILOVER_REMOTE_LOOKUP=0 HICACHE_STORAGE_PREFETCH_POLICY=best_effort REMOTE_BACKUP_REQUEST_AWARE=1 REMOTE_BACKUP_BUFFER_GB=16 NUM_REQUESTS=4 MAX_TOKENS=512 PROMPT_REPEAT=256 UNIQUE_REQUEST_PREFIX=0 FAILURE_DELAY_S=12 READY_TIMEOUT_S=2400 REQUEST_TIMEOUT_S=900 sbatch slurm/aegiskv_dp2_failure_smoke.sbatch'
```

### strict_final_same_write_best_effort_r2_host_invisible_lookup_disabled

```bash
ssh hkbugpumanode 'cd ~/aegiskv_sglang_small && RUN_LABEL=strict_final_same_write_best_effort_r2_host_invisible_lookup_disabled PORT=32822 REMOTE_BACKUP_PORT=33822 ENABLE_REMOTE_BACKUP=1 DISABLE_FAILOVER_REMOTE_LOOKUP=1 HICACHE_STORAGE_PREFETCH_POLICY=best_effort REMOTE_BACKUP_REQUEST_AWARE=1 REMOTE_BACKUP_BUFFER_GB=16 NUM_REQUESTS=4 MAX_TOKENS=512 PROMPT_REPEAT=256 UNIQUE_REQUEST_PREFIX=0 FAILURE_DELAY_S=12 READY_TIMEOUT_S=2400 REQUEST_TIMEOUT_S=900 sbatch slurm/aegiskv_dp2_failure_smoke.sbatch'
```

### strict_final_same_write_best_effort_r2_host_visible_lookup_enabled

```bash
ssh hkbugpumanode 'cd ~/aegiskv_sglang_small && RUN_LABEL=strict_final_same_write_best_effort_r2_host_visible_lookup_enabled PORT=32823 REMOTE_BACKUP_PORT=33823 ENABLE_REMOTE_BACKUP=1 DISABLE_FAILOVER_REMOTE_LOOKUP=0 HICACHE_STORAGE_PREFETCH_POLICY=best_effort REMOTE_BACKUP_REQUEST_AWARE=1 REMOTE_BACKUP_BUFFER_GB=16 NUM_REQUESTS=4 MAX_TOKENS=512 PROMPT_REPEAT=256 UNIQUE_REQUEST_PREFIX=0 FAILURE_DELAY_S=12 READY_TIMEOUT_S=2400 REQUEST_TIMEOUT_S=900 sbatch slurm/aegiskv_dp2_failure_smoke.sbatch'
```

### strict_final_same_write_best_effort_r3_host_invisible_lookup_disabled

```bash
ssh hkbugpumanode 'cd ~/aegiskv_sglang_small && RUN_LABEL=strict_final_same_write_best_effort_r3_host_invisible_lookup_disabled PORT=32824 REMOTE_BACKUP_PORT=33824 ENABLE_REMOTE_BACKUP=1 DISABLE_FAILOVER_REMOTE_LOOKUP=1 HICACHE_STORAGE_PREFETCH_POLICY=best_effort REMOTE_BACKUP_REQUEST_AWARE=1 REMOTE_BACKUP_BUFFER_GB=16 NUM_REQUESTS=4 MAX_TOKENS=512 PROMPT_REPEAT=256 UNIQUE_REQUEST_PREFIX=0 FAILURE_DELAY_S=12 READY_TIMEOUT_S=2400 REQUEST_TIMEOUT_S=900 sbatch slurm/aegiskv_dp2_failure_smoke.sbatch'
```

### strict_final_same_write_best_effort_r3_host_visible_lookup_enabled

```bash
ssh hkbugpumanode 'cd ~/aegiskv_sglang_small && RUN_LABEL=strict_final_same_write_best_effort_r3_host_visible_lookup_enabled PORT=32825 REMOTE_BACKUP_PORT=33825 ENABLE_REMOTE_BACKUP=1 DISABLE_FAILOVER_REMOTE_LOOKUP=0 HICACHE_STORAGE_PREFETCH_POLICY=best_effort REMOTE_BACKUP_REQUEST_AWARE=1 REMOTE_BACKUP_BUFFER_GB=16 NUM_REQUESTS=4 MAX_TOKENS=512 PROMPT_REPEAT=256 UNIQUE_REQUEST_PREFIX=0 FAILURE_DELAY_S=12 READY_TIMEOUT_S=2400 REQUEST_TIMEOUT_S=900 sbatch slurm/aegiskv_dp2_failure_smoke.sbatch'
```

### strict_final_same_write_wait_complete_r1_host_invisible_lookup_disabled

```bash
ssh hkbugpumanode 'cd ~/aegiskv_sglang_small && RUN_LABEL=strict_final_same_write_wait_complete_r1_host_invisible_lookup_disabled PORT=32826 REMOTE_BACKUP_PORT=33826 ENABLE_REMOTE_BACKUP=1 DISABLE_FAILOVER_REMOTE_LOOKUP=1 HICACHE_STORAGE_PREFETCH_POLICY=wait_complete REMOTE_BACKUP_REQUEST_AWARE=1 REMOTE_BACKUP_BUFFER_GB=16 NUM_REQUESTS=4 MAX_TOKENS=512 PROMPT_REPEAT=256 UNIQUE_REQUEST_PREFIX=0 FAILURE_DELAY_S=12 READY_TIMEOUT_S=2400 REQUEST_TIMEOUT_S=900 sbatch slurm/aegiskv_dp2_failure_smoke.sbatch'
```

### strict_final_same_write_wait_complete_r1_host_visible_lookup_enabled

```bash
ssh hkbugpumanode 'cd ~/aegiskv_sglang_small && RUN_LABEL=strict_final_same_write_wait_complete_r1_host_visible_lookup_enabled PORT=32827 REMOTE_BACKUP_PORT=33827 ENABLE_REMOTE_BACKUP=1 DISABLE_FAILOVER_REMOTE_LOOKUP=0 HICACHE_STORAGE_PREFETCH_POLICY=wait_complete REMOTE_BACKUP_REQUEST_AWARE=1 REMOTE_BACKUP_BUFFER_GB=16 NUM_REQUESTS=4 MAX_TOKENS=512 PROMPT_REPEAT=256 UNIQUE_REQUEST_PREFIX=0 FAILURE_DELAY_S=12 READY_TIMEOUT_S=2400 REQUEST_TIMEOUT_S=900 sbatch slurm/aegiskv_dp2_failure_smoke.sbatch'
```

### strict_final_same_write_wait_complete_r2_host_invisible_lookup_disabled

```bash
ssh hkbugpumanode 'cd ~/aegiskv_sglang_small && RUN_LABEL=strict_final_same_write_wait_complete_r2_host_invisible_lookup_disabled PORT=32828 REMOTE_BACKUP_PORT=33828 ENABLE_REMOTE_BACKUP=1 DISABLE_FAILOVER_REMOTE_LOOKUP=1 HICACHE_STORAGE_PREFETCH_POLICY=wait_complete REMOTE_BACKUP_REQUEST_AWARE=1 REMOTE_BACKUP_BUFFER_GB=16 NUM_REQUESTS=4 MAX_TOKENS=512 PROMPT_REPEAT=256 UNIQUE_REQUEST_PREFIX=0 FAILURE_DELAY_S=12 READY_TIMEOUT_S=2400 REQUEST_TIMEOUT_S=900 sbatch slurm/aegiskv_dp2_failure_smoke.sbatch'
```

### strict_final_same_write_wait_complete_r2_host_visible_lookup_enabled

```bash
ssh hkbugpumanode 'cd ~/aegiskv_sglang_small && RUN_LABEL=strict_final_same_write_wait_complete_r2_host_visible_lookup_enabled PORT=32829 REMOTE_BACKUP_PORT=33829 ENABLE_REMOTE_BACKUP=1 DISABLE_FAILOVER_REMOTE_LOOKUP=0 HICACHE_STORAGE_PREFETCH_POLICY=wait_complete REMOTE_BACKUP_REQUEST_AWARE=1 REMOTE_BACKUP_BUFFER_GB=16 NUM_REQUESTS=4 MAX_TOKENS=512 PROMPT_REPEAT=256 UNIQUE_REQUEST_PREFIX=0 FAILURE_DELAY_S=12 READY_TIMEOUT_S=2400 REQUEST_TIMEOUT_S=900 sbatch slurm/aegiskv_dp2_failure_smoke.sbatch'
```

### strict_final_same_write_wait_complete_r3_host_invisible_lookup_disabled

```bash
ssh hkbugpumanode 'cd ~/aegiskv_sglang_small && RUN_LABEL=strict_final_same_write_wait_complete_r3_host_invisible_lookup_disabled PORT=32830 REMOTE_BACKUP_PORT=33830 ENABLE_REMOTE_BACKUP=1 DISABLE_FAILOVER_REMOTE_LOOKUP=1 HICACHE_STORAGE_PREFETCH_POLICY=wait_complete REMOTE_BACKUP_REQUEST_AWARE=1 REMOTE_BACKUP_BUFFER_GB=16 NUM_REQUESTS=4 MAX_TOKENS=512 PROMPT_REPEAT=256 UNIQUE_REQUEST_PREFIX=0 FAILURE_DELAY_S=12 READY_TIMEOUT_S=2400 REQUEST_TIMEOUT_S=900 sbatch slurm/aegiskv_dp2_failure_smoke.sbatch'
```

### strict_final_same_write_wait_complete_r3_host_visible_lookup_enabled

```bash
ssh hkbugpumanode 'cd ~/aegiskv_sglang_small && RUN_LABEL=strict_final_same_write_wait_complete_r3_host_visible_lookup_enabled PORT=32831 REMOTE_BACKUP_PORT=33831 ENABLE_REMOTE_BACKUP=1 DISABLE_FAILOVER_REMOTE_LOOKUP=0 HICACHE_STORAGE_PREFETCH_POLICY=wait_complete REMOTE_BACKUP_REQUEST_AWARE=1 REMOTE_BACKUP_BUFFER_GB=16 NUM_REQUESTS=4 MAX_TOKENS=512 PROMPT_REPEAT=256 UNIQUE_REQUEST_PREFIX=0 FAILURE_DELAY_S=12 READY_TIMEOUT_S=2400 REQUEST_TIMEOUT_S=900 sbatch slurm/aegiskv_dp2_failure_smoke.sbatch'
```

## Pair Sync And Parsing

Replace each `<..._job_id>` placeholder with the Slurm job id returned by
`sbatch`.

### strict_final_same_write_best_effort_r1

Acceptance criteria:

- both runs have success_count == request_count
- both runs have pre_failover_backed_up_tokens > 0
- visible run has max_data_pages > 0 and prefetch_accepted_count > 0
- visible run reaches partial_async_materialized or full_prefetch_materialized
- visible_retry_latency_norm_vs_invisible < 1.0 for most repeats

```bash
python /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/scripts/sync_failure_pair_from_remote.py --invisible-job-id '<strict_final_same_write_best_effort_r1_host_invisible_lookup_disabled_job_id>' --visible-job-id '<strict_final_same_write_best_effort_r1_host_visible_lookup_enabled_job_id>' --run-id strict_final_same_write_best_effort_r1 --same-write-cost-control --prefetch-policy best_effort --title 'Strict-final recovery pair strict_final_same_write_best_effort_r1' --note 'Both runs pay remote_backup write cost; only the visible run can look up Host pages during failover. This is the recovery-benefit candidate.'
python /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/scripts/extract_physical_restore_evidence.py --result-dir 'invisible=/home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/dp2_failure_smoke_<strict_final_same_write_best_effort_r1_host_invisible_lookup_disabled_job_id>' --result-dir 'visible=/home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/dp2_failure_smoke_<strict_final_same_write_best_effort_r1_host_visible_lookup_enabled_job_id>' --output-csv /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/physical_restore_evidence/strict_final_same_write_best_effort_r1/physical_restore_runs.csv --output-json /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/physical_restore_evidence/strict_final_same_write_best_effort_r1/physical_restore_runs.json --output-md /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/physical_restore_evidence/strict_final_same_write_best_effort_r1/physical_restore_runs.md
```

### strict_final_same_write_best_effort_r2

Acceptance criteria:

- both runs have success_count == request_count
- both runs have pre_failover_backed_up_tokens > 0
- visible run has max_data_pages > 0 and prefetch_accepted_count > 0
- visible run reaches partial_async_materialized or full_prefetch_materialized
- visible_retry_latency_norm_vs_invisible < 1.0 for most repeats

```bash
python /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/scripts/sync_failure_pair_from_remote.py --invisible-job-id '<strict_final_same_write_best_effort_r2_host_invisible_lookup_disabled_job_id>' --visible-job-id '<strict_final_same_write_best_effort_r2_host_visible_lookup_enabled_job_id>' --run-id strict_final_same_write_best_effort_r2 --same-write-cost-control --prefetch-policy best_effort --title 'Strict-final recovery pair strict_final_same_write_best_effort_r2' --note 'Both runs pay remote_backup write cost; only the visible run can look up Host pages during failover. This is the recovery-benefit candidate.'
python /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/scripts/extract_physical_restore_evidence.py --result-dir 'invisible=/home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/dp2_failure_smoke_<strict_final_same_write_best_effort_r2_host_invisible_lookup_disabled_job_id>' --result-dir 'visible=/home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/dp2_failure_smoke_<strict_final_same_write_best_effort_r2_host_visible_lookup_enabled_job_id>' --output-csv /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/physical_restore_evidence/strict_final_same_write_best_effort_r2/physical_restore_runs.csv --output-json /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/physical_restore_evidence/strict_final_same_write_best_effort_r2/physical_restore_runs.json --output-md /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/physical_restore_evidence/strict_final_same_write_best_effort_r2/physical_restore_runs.md
```

### strict_final_same_write_best_effort_r3

Acceptance criteria:

- both runs have success_count == request_count
- both runs have pre_failover_backed_up_tokens > 0
- visible run has max_data_pages > 0 and prefetch_accepted_count > 0
- visible run reaches partial_async_materialized or full_prefetch_materialized
- visible_retry_latency_norm_vs_invisible < 1.0 for most repeats

```bash
python /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/scripts/sync_failure_pair_from_remote.py --invisible-job-id '<strict_final_same_write_best_effort_r3_host_invisible_lookup_disabled_job_id>' --visible-job-id '<strict_final_same_write_best_effort_r3_host_visible_lookup_enabled_job_id>' --run-id strict_final_same_write_best_effort_r3 --same-write-cost-control --prefetch-policy best_effort --title 'Strict-final recovery pair strict_final_same_write_best_effort_r3' --note 'Both runs pay remote_backup write cost; only the visible run can look up Host pages during failover. This is the recovery-benefit candidate.'
python /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/scripts/extract_physical_restore_evidence.py --result-dir 'invisible=/home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/dp2_failure_smoke_<strict_final_same_write_best_effort_r3_host_invisible_lookup_disabled_job_id>' --result-dir 'visible=/home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/dp2_failure_smoke_<strict_final_same_write_best_effort_r3_host_visible_lookup_enabled_job_id>' --output-csv /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/physical_restore_evidence/strict_final_same_write_best_effort_r3/physical_restore_runs.csv --output-json /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/physical_restore_evidence/strict_final_same_write_best_effort_r3/physical_restore_runs.json --output-md /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/physical_restore_evidence/strict_final_same_write_best_effort_r3/physical_restore_runs.md
```

### strict_final_same_write_wait_complete_r1

Acceptance criteria:

- both runs have success_count == request_count
- both runs have pre_failover_backed_up_tokens > 0
- visible run reaches physical_restore_state == full_prefetch_materialized
- visible run has prefetch_io_completion_ratio == 1.0

```bash
python /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/scripts/sync_failure_pair_from_remote.py --invisible-job-id '<strict_final_same_write_wait_complete_r1_host_invisible_lookup_disabled_job_id>' --visible-job-id '<strict_final_same_write_wait_complete_r1_host_visible_lookup_enabled_job_id>' --run-id strict_final_same_write_wait_complete_r1 --same-write-cost-control --prefetch-policy wait_complete --title 'Strict-final recovery pair strict_final_same_write_wait_complete_r1' --note 'Both runs pay remote_backup write cost; wait_complete makes the full Host restore path auditable even if it is slower.'
python /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/scripts/extract_physical_restore_evidence.py --result-dir 'invisible=/home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/dp2_failure_smoke_<strict_final_same_write_wait_complete_r1_host_invisible_lookup_disabled_job_id>' --result-dir 'visible=/home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/dp2_failure_smoke_<strict_final_same_write_wait_complete_r1_host_visible_lookup_enabled_job_id>' --output-csv /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/physical_restore_evidence/strict_final_same_write_wait_complete_r1/physical_restore_runs.csv --output-json /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/physical_restore_evidence/strict_final_same_write_wait_complete_r1/physical_restore_runs.json --output-md /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/physical_restore_evidence/strict_final_same_write_wait_complete_r1/physical_restore_runs.md
```

### strict_final_same_write_wait_complete_r2

Acceptance criteria:

- both runs have success_count == request_count
- both runs have pre_failover_backed_up_tokens > 0
- visible run reaches physical_restore_state == full_prefetch_materialized
- visible run has prefetch_io_completion_ratio == 1.0

```bash
python /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/scripts/sync_failure_pair_from_remote.py --invisible-job-id '<strict_final_same_write_wait_complete_r2_host_invisible_lookup_disabled_job_id>' --visible-job-id '<strict_final_same_write_wait_complete_r2_host_visible_lookup_enabled_job_id>' --run-id strict_final_same_write_wait_complete_r2 --same-write-cost-control --prefetch-policy wait_complete --title 'Strict-final recovery pair strict_final_same_write_wait_complete_r2' --note 'Both runs pay remote_backup write cost; wait_complete makes the full Host restore path auditable even if it is slower.'
python /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/scripts/extract_physical_restore_evidence.py --result-dir 'invisible=/home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/dp2_failure_smoke_<strict_final_same_write_wait_complete_r2_host_invisible_lookup_disabled_job_id>' --result-dir 'visible=/home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/dp2_failure_smoke_<strict_final_same_write_wait_complete_r2_host_visible_lookup_enabled_job_id>' --output-csv /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/physical_restore_evidence/strict_final_same_write_wait_complete_r2/physical_restore_runs.csv --output-json /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/physical_restore_evidence/strict_final_same_write_wait_complete_r2/physical_restore_runs.json --output-md /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/physical_restore_evidence/strict_final_same_write_wait_complete_r2/physical_restore_runs.md
```

### strict_final_same_write_wait_complete_r3

Acceptance criteria:

- both runs have success_count == request_count
- both runs have pre_failover_backed_up_tokens > 0
- visible run reaches physical_restore_state == full_prefetch_materialized
- visible run has prefetch_io_completion_ratio == 1.0

```bash
python /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/scripts/sync_failure_pair_from_remote.py --invisible-job-id '<strict_final_same_write_wait_complete_r3_host_invisible_lookup_disabled_job_id>' --visible-job-id '<strict_final_same_write_wait_complete_r3_host_visible_lookup_enabled_job_id>' --run-id strict_final_same_write_wait_complete_r3 --same-write-cost-control --prefetch-policy wait_complete --title 'Strict-final recovery pair strict_final_same_write_wait_complete_r3' --note 'Both runs pay remote_backup write cost; wait_complete makes the full Host restore path auditable even if it is slower.'
python /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/scripts/extract_physical_restore_evidence.py --result-dir 'invisible=/home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/dp2_failure_smoke_<strict_final_same_write_wait_complete_r3_host_invisible_lookup_disabled_job_id>' --result-dir 'visible=/home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/dp2_failure_smoke_<strict_final_same_write_wait_complete_r3_host_visible_lookup_enabled_job_id>' --output-csv /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/physical_restore_evidence/strict_final_same_write_wait_complete_r3/physical_restore_runs.csv --output-json /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/physical_restore_evidence/strict_final_same_write_wait_complete_r3/physical_restore_runs.json --output-md /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/physical_restore_evidence/strict_final_same_write_wait_complete_r3/physical_restore_runs.md
```

## Aggregate

```bash
python /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/scripts/aggregate_failure_visibility_pairs.py --results-dir /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results --output-dir /home/comp/zhanzhao/sglsim/benchmark/AegisKV/presentation/experiments/sglang_small/results/failure_visibility_pair_aggregate
```

## Boundary

These runs are a real-system recovery-path matrix. They do not by themselves turn SGLSim restore rows into physical_tensor_copy rows; the P0 audit should only pass after real recovery evidence is connected to the final policy claims.
