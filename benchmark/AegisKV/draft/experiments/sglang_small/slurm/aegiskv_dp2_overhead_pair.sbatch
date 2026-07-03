#!/usr/bin/env bash
#SBATCH --job-name=aegiskv-dp2-overhead-pair
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --time=01:20:00
#SBATCH --output=results/slurm-%x-%j.out
#SBATCH --error=results/slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SGLANG_DIR="${SGLANG_DIR:-/home/comp/25482351/sglang}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
BASE_PORT="${PORT:-$((31000 + (${SLURM_JOB_ID:-1} % 800)))}"
OUT_DIR="${ROOT_DIR}/results/dp2_overhead_pair_${SLURM_JOB_ID:-manual}"
UV_BIN="${UV_BIN:-/home/comp/25482351/.local/bin/uv}"
NUM_REQUESTS="${NUM_REQUESTS:-8}"
MAX_TOKENS="${MAX_TOKENS:-64}"

mkdir -p "${OUT_DIR}"
cd "${SGLANG_DIR}"

export PYTHONPATH="${SGLANG_DIR}/python:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"

{
  echo "timestamp=$(date -Iseconds)"
  echo "host=$(hostname)"
  echo "user=$(id -un)"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "sglang_dir=${SGLANG_DIR}"
  echo "model_path=${MODEL_PATH}"
  echo "base_port=${BASE_PORT}"
  echo "num_requests=${NUM_REQUESTS}"
  echo "max_tokens=${MAX_TOKENS}"
  echo "no_proxy=${no_proxy:-}"
  git branch --show-current || true
  git rev-parse HEAD || true
  git status --short || true
} | tee "${OUT_DIR}/provenance.txt"

SERVER_PID=""

cleanup() {
  if [ -n "${SERVER_PID:-}" ] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

run_case() {
  local case_name="$1"
  local port="$2"
  shift 2
  local case_dir="${OUT_DIR}/${case_name}"
  mkdir -p "${case_dir}"

  {
    echo "case=${case_name}"
    echo "port=${port}"
    echo "extra_server_args=$*"
    echo "case_start=$(date -Iseconds)"
  } | tee "${case_dir}/case_provenance.txt"

  SERVER_LOG="${case_dir}/server.log"
  CLIENT_JSONL="${case_dir}/client_results.jsonl"
  SUMMARY_JSON="${case_dir}/summary.json"

  "${UV_BIN}" run python -u -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --tp-size 1 \
    --dp-size 2 \
    --load-balance-method round_robin \
    --attention-backend triton \
    --sampling-backend pytorch \
    --disable-cuda-graph \
    --disable-piecewise-cuda-graph \
    "$@" \
    > "${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!

  "${UV_BIN}" run python "${ROOT_DIR}/scripts/sglang_smoke_client.py" \
    --host 127.0.0.1 \
    --port "${port}" \
    --model "${MODEL_PATH}" \
    --num-requests "${NUM_REQUESTS}" \
    --max-tokens "${MAX_TOKENS}" \
    --output-jsonl "${CLIENT_JSONL}" \
    --summary-json "${SUMMARY_JSON}"

  "${UV_BIN}" run python "${ROOT_DIR}/scripts/summarize_smoke.py" \
    --input-jsonl "${CLIENT_JSONL}" \
    --output-dir "${case_dir}"

  cleanup
  SERVER_PID=""
  sleep 8
  echo "case_end=$(date -Iseconds)" | tee -a "${case_dir}/case_provenance.txt"
}

run_case "baseline_no_hicache" "${BASE_PORT}"
run_case "hicache_write_through" "$((BASE_PORT + 1))" \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-size 0 \
  --hicache-write-policy write_through

"${UV_BIN}" run python "${ROOT_DIR}/scripts/summarize_overhead_pair.py" \
  --baseline-dir "${OUT_DIR}/baseline_no_hicache" \
  --hicache-dir "${OUT_DIR}/hicache_write_through" \
  --output-dir "${OUT_DIR}" \
  --run-id "dp2_overhead_pair_${SLURM_JOB_ID:-manual}"

echo "wrote ${OUT_DIR}"
