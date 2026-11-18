#!/usr/bin/env bash
#SBATCH --job-name=aegiskv-dp2-failure-smoke
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --exclude=hkbugpusrv07
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm-%x-%j.out
#SBATCH --error=results/slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SGLANG_DIR="${SGLANG_DIR:-/home/comp/25482351/sglang}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
PORT="${PORT:-$((32000 + (${SLURM_JOB_ID:-1} % 800)))}"
OUT_DIR="${ROOT_DIR}/results/dp2_failure_smoke_${SLURM_JOB_ID:-manual}"
UV_BIN="${UV_BIN:-/home/comp/25482351/.local/bin/uv}"
PYTHON_BIN="${PYTHON_BIN:-${SGLANG_DIR}/.venv/bin/python3}"
NUM_REQUESTS="${NUM_REQUESTS:-4}"
MAX_TOKENS="${MAX_TOKENS:-256}"
PROMPT_REPEAT="${PROMPT_REPEAT:-64}"
UNIQUE_REQUEST_PREFIX="${UNIQUE_REQUEST_PREFIX:-0}"
FAIL_DP_RANK="${FAIL_DP_RANK:-0}"
FAILURE_DELAY_S="${FAILURE_DELAY_S:-4}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-900}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-900}"
DISABLE_OVERLAP_SCHEDULE="${DISABLE_OVERLAP_SCHEDULE:-1}"
ENABLE_REMOTE_BACKUP="${ENABLE_REMOTE_BACKUP:-0}"
DISABLE_FAILOVER_REMOTE_LOOKUP="${DISABLE_FAILOVER_REMOTE_LOOKUP:-0}"
REMOTE_BACKUP_PORT="${REMOTE_BACKUP_PORT:-$((33000 + (${SLURM_JOB_ID:-1} % 800)))}"
REMOTE_BACKUP_BUFFER_GB="${REMOTE_BACKUP_BUFFER_GB:-16}"
REMOTE_BACKUP_REQUEST_AWARE="${REMOTE_BACKUP_REQUEST_AWARE:-1}"
HICACHE_STORAGE_PREFETCH_POLICY="${HICACHE_STORAGE_PREFETCH_POLICY:-best_effort}"

mkdir -p "${OUT_DIR}"
cd "${SGLANG_DIR}"

if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="${SGLANG_DIR}/.venv/bin/python"
fi
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Could not find executable SGLang Python at ${SGLANG_DIR}/.venv/bin/python3 or python" >&2
  exit 2
fi

export PYTHONPATH="${SGLANG_DIR}/python:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export UV_NO_SYNC="${UV_NO_SYNC:-1}"
export PATH="${SGLANG_DIR}/.venv/bin:/home/comp/25482351/.local/bin:${PATH}"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export SGLANG_FAILOVER_EVENTS_FILE="${OUT_DIR}/failover_events.jsonl"
export SGLANG_FAILOVER_DISABLE_REMOTE_LOOKUP="${DISABLE_FAILOVER_REMOTE_LOOKUP}"

{
  echo "timestamp=$(date -Iseconds)"
  echo "host=$(hostname)"
  echo "user=$(id -un)"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "sglang_dir=${SGLANG_DIR}"
  echo "python_bin=${PYTHON_BIN}"
  echo "uv_bin=${UV_BIN}"
  echo "uv_no_sync=${UV_NO_SYNC}"
  echo "path=${PATH}"
  echo "ninja=$(command -v ninja || true)"
  echo "model_path=${MODEL_PATH}"
  echo "port=${PORT}"
  echo "num_requests=${NUM_REQUESTS}"
  echo "max_tokens=${MAX_TOKENS}"
  echo "prompt_repeat=${PROMPT_REPEAT}"
  echo "unique_request_prefix=${UNIQUE_REQUEST_PREFIX}"
  echo "fail_dp_rank=${FAIL_DP_RANK}"
  echo "failure_delay_s=${FAILURE_DELAY_S}"
  echo "ready_timeout_s=${READY_TIMEOUT_S}"
  echo "request_timeout_s=${REQUEST_TIMEOUT_S}"
  echo "disable_overlap_schedule=${DISABLE_OVERLAP_SCHEDULE}"
  echo "enable_remote_backup=${ENABLE_REMOTE_BACKUP}"
  echo "disable_failover_remote_lookup=${DISABLE_FAILOVER_REMOTE_LOOKUP}"
  echo "sglang_failover_disable_remote_lookup=${SGLANG_FAILOVER_DISABLE_REMOTE_LOOKUP}"
  echo "remote_backup_port=${REMOTE_BACKUP_PORT}"
  echo "remote_backup_buffer_gb=${REMOTE_BACKUP_BUFFER_GB}"
  echo "remote_backup_request_aware=${REMOTE_BACKUP_REQUEST_AWARE}"
  echo "hicache_storage_prefetch_policy=${HICACHE_STORAGE_PREFETCH_POLICY}"
  echo "failover_events_file=${SGLANG_FAILOVER_EVENTS_FILE}"
  echo "no_proxy=${no_proxy:-}"
  git branch --show-current || true
  git rev-parse HEAD || true
  git status --short || true
} | tee "${OUT_DIR}/provenance.txt"

SERVER_LOG="${OUT_DIR}/server.log"
REMOTE_BACKUP_LOG="${OUT_DIR}/remote_backup_server.log"
CLIENT_JSONL="${OUT_DIR}/client_results.jsonl"
SUMMARY_JSON="${OUT_DIR}/failure_summary.json"
SUMMARY_CSV="${OUT_DIR}/failure_summary.csv"
CANDIDATE_CSV="${OUT_DIR}/candidate_fig_failure_smoke_sglang.csv"
SERVER_PID=""
REMOTE_BACKUP_PID=""
SERVER_EXTRA_ARGS=()
CLIENT_EXTRA_ARGS=()
if [ "${DISABLE_OVERLAP_SCHEDULE}" = "1" ]; then
  SERVER_EXTRA_ARGS+=(--disable-overlap-schedule)
fi
if [ "${ENABLE_REMOTE_BACKUP}" = "1" ]; then
  REMOTE_BACKUP_CONFIG=$(printf '{"remote_backup_mode":"client","remote_backup_url":"localhost:%s","remote_backup_buffer_size_gb":%s}' "${REMOTE_BACKUP_PORT}" "${REMOTE_BACKUP_BUFFER_GB}")
  SERVER_EXTRA_ARGS+=(
    --hicache-storage-backend remote_backup
    --hicache-storage-prefetch-policy "${HICACHE_STORAGE_PREFETCH_POLICY}"
    --hicache-storage-backend-extra-config "${REMOTE_BACKUP_CONFIG}"
  )
fi
if [ "${UNIQUE_REQUEST_PREFIX}" = "1" ]; then
  CLIENT_EXTRA_ARGS+=(--unique-request-prefix)
fi

cleanup() {
  if [ -n "${SERVER_PID:-}" ] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
  if [ -n "${REMOTE_BACKUP_PID:-}" ] && kill -0 "${REMOTE_BACKUP_PID}" >/dev/null 2>&1; then
    kill "${REMOTE_BACKUP_PID}" >/dev/null 2>&1 || true
    wait "${REMOTE_BACKUP_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [ "${ENABLE_REMOTE_BACKUP}" = "1" ]; then
  echo "remote_backup_launch_start=$(date -Iseconds)" | tee -a "${OUT_DIR}/provenance.txt"
  "${PYTHON_BIN}" -u "${ROOT_DIR}/scripts/launch_remote_backup_server_for_smoke.py" \
    --sglang-dir "${SGLANG_DIR}" \
    --port "${REMOTE_BACKUP_PORT}" \
    --buffer-size-gb "${REMOTE_BACKUP_BUFFER_GB}" \
    --page-size 1 \
    --request-aware "${REMOTE_BACKUP_REQUEST_AWARE}" \
    > "${REMOTE_BACKUP_LOG}" 2>&1 &
  REMOTE_BACKUP_PID=$!
  echo "remote_backup_pid=${REMOTE_BACKUP_PID}" | tee -a "${OUT_DIR}/provenance.txt"
  sleep 2
  if ! kill -0 "${REMOTE_BACKUP_PID}" >/dev/null 2>&1; then
    echo "remote_backup_failed_before_server=$(date -Iseconds)" | tee -a "${OUT_DIR}/provenance.txt"
    tail -n 200 "${REMOTE_BACKUP_LOG}" || true
    exit 1
  fi
fi

echo "server_launch_start=$(date -Iseconds)" | tee -a "${OUT_DIR}/provenance.txt"
"${PYTHON_BIN}" -u -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tp-size 1 \
  --dp-size 2 \
  --load-balance-method round_robin \
  --attention-backend triton \
  --sampling-backend pytorch \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-size 0 \
  --hicache-write-policy write_through \
  "${SERVER_EXTRA_ARGS[@]}" \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "server_pid=${SERVER_PID}" | tee -a "${OUT_DIR}/provenance.txt"
sleep 2
if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
  echo "server_failed_before_client=$(date -Iseconds)" | tee -a "${OUT_DIR}/provenance.txt"
  tail -n 200 "${SERVER_LOG}" || true
  exit 1
fi

echo "client_launch_start=$(date -Iseconds)" | tee -a "${OUT_DIR}/provenance.txt"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/sglang_failure_client.py" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --model "${MODEL_PATH}" \
  --num-requests "${NUM_REQUESTS}" \
  --max-tokens "${MAX_TOKENS}" \
  --prompt-repeat "${PROMPT_REPEAT}" \
  --fail-dp-rank "${FAIL_DP_RANK}" \
  --failure-delay-s "${FAILURE_DELAY_S}" \
  --ready-timeout-s "${READY_TIMEOUT_S}" \
  --request-timeout-s "${REQUEST_TIMEOUT_S}" \
  --recover \
  "${CLIENT_EXTRA_ARGS[@]}" \
  --output-jsonl "${CLIENT_JSONL}" \
  --summary-json "${SUMMARY_JSON}" \
  --summary-csv "${SUMMARY_CSV}" \
  --candidate-csv "${CANDIDATE_CSV}" \
  --failover-events-jsonl "${SGLANG_FAILOVER_EVENTS_FILE}" \
  --run-id "dp2_failure_smoke_${SLURM_JOB_ID:-manual}"

echo "client_done=$(date -Iseconds)" | tee -a "${OUT_DIR}/provenance.txt"
echo "wrote ${OUT_DIR}"
