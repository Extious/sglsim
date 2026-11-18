#!/usr/bin/env bash
#SBATCH --job-name=aegiskv-qwen3-8b-pp2-prof
#SBATCH --partition=debug,long
#SBATCH --exclude=gpu14
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --gres=gpu:rtx4090:2
#SBATCH --time=00:30:00
#SBATCH --output=/home/comp/zhanzhao/sglsim/AegisKV/flowchart/ideas/evidence/sglang_profiler/pp2_qwen3_8b/slurm-%j.out
#SBATCH --error=/home/comp/zhanzhao/sglsim/AegisKV/flowchart/ideas/evidence/sglang_profiler/pp2_qwen3_8b/slurm-%j.err

set -euo pipefail

EXPERIMENT_DIR=/home/comp/zhanzhao/sglsim/AegisKV/flowchart/ideas/evidence/sglang_profiler/pp2_qwen3_8b
SGLANG_WORKTREE=/home/comp/zhanzhao/.tmp/sglang-profiler-pp2
SGLANG_ORIGINAL=/home/comp/zhanzhao/sglang
PYTHON_BIN=/home/comp/zhanzhao/sglang/.venv/bin/python
MODEL_DIR=/home/comp/zhanzhao/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
SERVER_PORT=$((20000 + SLURM_JOB_ID % 20000))
RUN_DIR="${EXPERIMENT_DIR}/runs/${SLURM_JOB_ID}"
PROFILE_DIR="${RUN_DIR}/profile"
STOP_FILE="${RUN_DIR}/stop-resource-sampling"

mkdir -p "${PROFILE_DIR}"
rm -f "${STOP_FILE}"

export PATH="${SGLANG_ORIGINAL}/.venv/bin:${PATH}"
: "${CUDA_VISIBLE_DEVICES:?Slurm did not publish allocated GPUs}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_PROFILE_V2=0
export PYTHONPATH="${EXPERIMENT_DIR}/..:${SGLANG_WORKTREE}/python"

{
  date --iso-8601=seconds
  hostname
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID}"
  printf 'slurm_partition=%s\n' "${SLURM_JOB_PARTITION}"
  printf 'slurm_job_gpus=%s\n' "${SLURM_JOB_GPUS:-}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'server_port=%s\n' "${SERVER_PORT}"
  printf 'sglang_worktree=%s\n' "${SGLANG_WORKTREE}"
  printf 'sglang_commit=%s\n' "$(git -C "${SGLANG_WORKTREE}" rev-parse HEAD)"
  printf 'sglang_profile_v2=%s\n' "${SGLANG_PROFILE_V2}"
  printf 'model_dir=%s\n' "${MODEL_DIR}"
  "${PYTHON_BIN}" --version
  nvidia-smi -L
  nvidia-smi topo -m
  nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,pci.bus_id --format=csv
} >"${RUN_DIR}/environment.txt" 2>&1

GPU_PROCESS_ROWS=$(nvidia-smi \
  --query-compute-apps=pid,gpu_uuid,process_name,used_memory \
  --format=csv,noheader)
if [[ -n "${GPU_PROCESS_ROWS}" ]]; then
  printf '%s\n' "${GPU_PROCESS_ROWS}" \
    >"${RUN_DIR}/gpu_processes_before_launch.txt"
  printf 'allocated GPUs are not empty before launch:\n' >&2
  printf '%s\n' "${GPU_PROCESS_ROWS}" >&2
  exit 20
fi
printf 'status=empty\n' >"${RUN_DIR}/gpu_processes_before_launch.txt"

"${PYTHON_BIN}" -m sglang.launch_server \
  --model-path "${MODEL_DIR}" \
  --host 127.0.0.1 \
  --port "${SERVER_PORT}" \
  --tp-size 1 \
  --pp-size 2 \
  --context-length 16384 \
  --chunked-prefill-size 1024 \
  --max-running-requests 32 \
  --mem-fraction-static 0.80 \
  --disable-cuda-graph \
  --disable-radix-cache \
  --enable-metrics \
  --enable-request-time-stats-logging \
  --attention-backend triton \
  --sampling-backend pytorch \
  >"${RUN_DIR}/server.stdout.log" \
  2>"${RUN_DIR}/server.stderr.log" &
SERVER_PID=$!

SAMPLER_PID=
DMON_PID=
stop_server() {
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
cleanup() {
  touch "${STOP_FILE}"
  if [[ -n "${SAMPLER_PID}" ]]; then
    wait "${SAMPLER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${DMON_PID}" ]] && kill -0 "${DMON_PID}" 2>/dev/null; then
    kill -TERM "${DMON_PID}" 2>/dev/null || true
    wait "${DMON_PID}" 2>/dev/null || true
  fi
  stop_server
}
trap cleanup EXIT INT TERM

SERVER_READY=0
for _ in $(seq 1 450); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    set +e
    wait "${SERVER_PID}"
    SERVER_RC=$?
    set -e
    {
      printf 'server exited before /health became ready\n'
      printf 'server_pid=%s\n' "${SERVER_PID}"
      printf 'server_exit_code=%s\n' "${SERVER_RC}"
      date --iso-8601=seconds
    } | tee "${RUN_DIR}/server_exit_before_ready.log" >&2
    if [[ "${SERVER_RC}" -eq 0 ]]; then
      exit 30
    fi
    exit "${SERVER_RC}"
  fi
  if curl --fail --silent --max-time 2 \
    "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null; then
    SERVER_READY=1
    break
  fi
  sleep 2
done
if [[ "${SERVER_READY}" -ne 1 ]]; then
  {
    printf 'server did not become healthy before the 900-second deadline\n'
    printf 'server_pid=%s\n' "${SERVER_PID}"
    date --iso-8601=seconds
  } | tee "${RUN_DIR}/server_health_timeout.log" >&2
  exit 31
fi

"${PYTHON_BIN}" "${EXPERIMENT_DIR}/sample_resources.py" \
  --server-pid "${SERVER_PID}" \
  --output "${RUN_DIR}/resource_samples.csv" \
  --stop-file "${STOP_FILE}" \
  --interval 0.2 \
  >"${RUN_DIR}/resource_sampler.log" 2>&1 &
SAMPLER_PID=$!

nvidia-smi dmon \
  --select t \
  --delay 1 \
  --options DT \
  --format csv,nounit \
  >"${RUN_DIR}/pcie_throughput_dmon.csv" 2>"${RUN_DIR}/pcie_dmon.stderr.log" &
DMON_PID=$!

"${PYTHON_BIN}" "${EXPERIMENT_DIR}/profile_load.py" \
  --url "http://127.0.0.1:${SERVER_PORT}" \
  --output-dir "${PROFILE_DIR}" \
  --profile-id "slurm-${SLURM_JOB_ID}" \
  --profile-prefix "aegiskv-qwen3-8b-pp2" \
  --chunked-prefill-size 1024 \
  --calibration-concurrency 2 4 6 8 12 16 24 32 40 48 \
  --calibration-waves 1 \
  --offered-rate-factors 0.75 0.85 0.90 0.95 1.00 1.05 \
  --offered-rate-duration 12 \
  --steady-load-duration 20 \
  --profile-concurrency 16 \
  --prefill-input-len 4096 \
  --prefill-output-len 16 \
  --decode-input-len 128 \
  --decode-output-len 128 \
  --min-decode-concurrency 8 \
  --combined-profile-steps 100 \
  --memory-profile-steps 4 \
  --memory-profile-concurrency 8 \
  2>&1 | tee "${RUN_DIR}/workload.log"

curl --fail --silent --show-error \
  "http://127.0.0.1:${SERVER_PORT}/get_server_info" \
  >"${RUN_DIR}/live_server_info.json"
curl --fail --silent --show-error \
  "http://127.0.0.1:${SERVER_PORT}/v1/models" \
  >"${RUN_DIR}/live_models.json"
curl --fail --silent --show-error \
  "http://127.0.0.1:${SERVER_PORT}/metrics" \
  >"${RUN_DIR}/metrics_after.prom"

touch "${STOP_FILE}"
wait "${SAMPLER_PID}" || true
SAMPLER_PID=
kill -TERM "${DMON_PID}" 2>/dev/null || true
wait "${DMON_PID}" 2>/dev/null || true
DMON_PID=

# Release the model workers and their HBM before CPU-heavy trace conversion,
# merging, and report generation. The Slurm allocation remains active until
# the verified artifacts and checksums are complete.
stop_server

"${PYTHON_BIN}" "${EXPERIMENT_DIR}/export_memory_html.py" \
  --root "${PROFILE_DIR}" \
  >"${RUN_DIR}/memory_html_files.txt"

for rank in 0 1; do
  mapfile -t COMBINED_TRACES < <(
    find "${PROFILE_DIR}/traces/combined" -type f \
      -name "*PP-${rank}.trace.json.gz" | sort
  )
  if [[ "${#COMBINED_TRACES[@]}" -ne 1 ]]; then
    printf 'rank %s expected one automatic combined trace, got %s\n' \
      "${rank}" "${#COMBINED_TRACES[@]}" >&2
    exit 4
  fi
  "${PYTHON_BIN}" "${EXPERIMENT_DIR}/../analyze_stage_timeline.py" \
    --prefill "${COMBINED_TRACES[0]}" \
    --decode "${COMBINED_TRACES[0]}" \
    --output "${RUN_DIR}/stage_timeline_pp${rank}.json"

  "${PYTHON_BIN}" \
    "${SGLANG_WORKTREE}/.claude/skills/llm-torch-profiler-analysis/scripts/analyze_llm_torch_profile.py" \
    --framework sglang \
    --input "${COMBINED_TRACES[0]}" \
    --kernel-table-limit 30 \
    --overlap-table-limit 30 \
    >"${RUN_DIR}/triage_combined_pp${rank}.md"
done

"${PYTHON_BIN}" "${EXPERIMENT_DIR}/build_browser_trace.py" \
  --run-dir "${RUN_DIR}" \
  >"${RUN_DIR}/browser_trace_build.log"

"${PYTHON_BIN}" "${EXPERIMENT_DIR}/analyze_run.py" \
  --run-dir "${RUN_DIR}"

find "${RUN_DIR}" -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  >"${RUN_DIR}/SHA256SUMS"

printf '%s\n' "${RUN_DIR}" >"${EXPERIMENT_DIR}/LATEST_RUN"
