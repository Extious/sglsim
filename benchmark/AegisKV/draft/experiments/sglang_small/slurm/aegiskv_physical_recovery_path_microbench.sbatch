#!/usr/bin/env bash
#SBATCH --job-name=aegiskv-kvrecovery
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --exclude=hkbugpusrv07
#SBATCH --time=00:20:00
#SBATCH --output=results/slurm-%x-%j.out
#SBATCH --error=results/slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SGLANG_DIR="${SGLANG_DIR:-/home/comp/25482351/sglang}"
PYTHON_BIN="${PYTHON_BIN:-${SGLANG_DIR}/.venv/bin/python3}"
OUT_DIR="${ROOT_DIR}/results/physical_recovery_path_${SLURM_JOB_ID:-manual}"
TOKEN_COUNTS="${TOKEN_COUNTS:-512,1024,2048,4096,8192}"
BYTES_PER_TOKEN="${BYTES_PER_TOKEN:-147456}"
REPEATS="${REPEATS:-20}"
WARMUP="${WARMUP:-5}"
DTYPE="${DTYPE:-bfloat16}"
SOURCE_DEVICE="${SOURCE_DEVICE:-0}"
TARGET_DEVICE="${TARGET_DEVICE:-1}"

mkdir -p "${OUT_DIR}"

if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="${SGLANG_DIR}/.venv/bin/python"
fi
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Could not find executable SGLang Python at ${SGLANG_DIR}/.venv/bin/python3 or python" >&2
  exit 2
fi

{
  echo "timestamp=$(date -Iseconds)"
  echo "host=$(hostname)"
  echo "user=$(id -un)"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "slurm_partition=${SLURM_JOB_PARTITION:-}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "sglang_dir=${SGLANG_DIR}"
  echo "python_bin=${PYTHON_BIN}"
  echo "token_counts=${TOKEN_COUNTS}"
  echo "bytes_per_token=${BYTES_PER_TOKEN}"
  echo "repeats=${REPEATS}"
  echo "warmup=${WARMUP}"
  echo "dtype=${DTYPE}"
  echo "source_device=${SOURCE_DEVICE}"
  echo "target_device=${TARGET_DEVICE}"
  if [ -d "${SGLANG_DIR}/.git" ]; then
    git -C "${SGLANG_DIR}" branch --show-current || true
    git -C "${SGLANG_DIR}" rev-parse HEAD || true
  fi
} | tee "${OUT_DIR}/provenance.txt"

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L > "${OUT_DIR}/nvidia-smi-L.txt" || true
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi topo -m > "${OUT_DIR}/nvidia-smi-topo.txt" || true
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi > "${OUT_DIR}/nvidia-smi.txt" || true

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SGLANG_DIR}/python:${PYTHONPATH:-}"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/physical_recovery_path_microbench.py" \
  --token-counts "${TOKEN_COUNTS}" \
  --bytes-per-token "${BYTES_PER_TOKEN}" \
  --repeats "${REPEATS}" \
  --warmup "${WARMUP}" \
  --dtype "${DTYPE}" \
  --source-device "${SOURCE_DEVICE}" \
  --target-device "${TARGET_DEVICE}" \
  --output-dir "${OUT_DIR}"

echo "wrote ${OUT_DIR}"
