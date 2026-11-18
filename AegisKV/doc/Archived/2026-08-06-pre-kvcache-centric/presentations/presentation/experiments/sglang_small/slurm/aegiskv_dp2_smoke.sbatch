#!/usr/bin/env bash
#SBATCH --job-name=aegiskv-dp2-smoke
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --time=00:45:00
#SBATCH --output=results/slurm-%x-%j.out
#SBATCH --error=results/slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SGLANG_DIR="${SGLANG_DIR:-/home/comp/25482351/sglang}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
PORT="${PORT:-$((30000 + (${SLURM_JOB_ID:-1} % 1000)))}"
OUT_DIR="${ROOT_DIR}/results/dp2_smoke_${SLURM_JOB_ID:-manual}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python}"
UV_BIN="${UV_BIN:-/home/comp/25482351/.local/bin/uv}"

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
  echo "port=${PORT}"
  echo "no_proxy=${no_proxy:-}"
  git branch --show-current || true
  git rev-parse HEAD || true
} | tee "${OUT_DIR}/provenance.txt"

SERVER_LOG="${OUT_DIR}/server.log"
CLIENT_JSONL="${OUT_DIR}/client_results.jsonl"
SUMMARY_JSON="${OUT_DIR}/summary.json"

cleanup() {
  if [ -n "${SERVER_PID:-}" ] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

"${UV_BIN}" run python -u -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tp-size 1 \
  --dp-size 2 \
  --load-balance-method round_robin \
  --attention-backend triton \
  --sampling-backend pytorch \
  --disable-cuda-graph \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-size 0 \
  --hicache-write-policy write_through \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

"${UV_BIN}" run python "${ROOT_DIR}/scripts/sglang_smoke_client.py" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --model "${MODEL_PATH}" \
  --num-requests 8 \
  --max-tokens 64 \
  --output-jsonl "${CLIENT_JSONL}" \
  --summary-json "${SUMMARY_JSON}"

"${UV_BIN}" run python "${ROOT_DIR}/scripts/summarize_smoke.py" \
  --input-jsonl "${CLIENT_JSONL}" \
  --output-dir "${OUT_DIR}"

echo "wrote ${OUT_DIR}"
