#!/usr/bin/env bash
#SBATCH --job-name=aegiskv-env
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:10:00
#SBATCH --output=results/slurm-%x-%j.out
#SBATCH --error=results/slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
OUT_DIR="${ROOT_DIR}/results/env_${SLURM_JOB_ID:-manual}"
SGLANG_DIR="${SGLANG_DIR:-/home/comp/25482351/sglang}"

mkdir -p "${OUT_DIR}"

{
  echo "timestamp=$(date -Iseconds)"
  echo "host=$(hostname)"
  echo "user=$(id -un)"
  echo "pwd=${PWD}"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "slurm_partition=${SLURM_JOB_PARTITION:-}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "sglang_dir=${SGLANG_DIR}"
} | tee "${OUT_DIR}/env.txt"

command -v sinfo >/dev/null 2>&1 && sinfo -h -o "%P|%D|%t|%G" > "${OUT_DIR}/sinfo.txt" || true
command -v squeue >/dev/null 2>&1 && squeue -j "${SLURM_JOB_ID:-0}" > "${OUT_DIR}/squeue.txt" || true
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L > "${OUT_DIR}/nvidia-smi-L.txt" || true
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi > "${OUT_DIR}/nvidia-smi.txt" || true

if [ -d "${SGLANG_DIR}/.git" ]; then
  git -C "${SGLANG_DIR}" branch --show-current > "${OUT_DIR}/sglang_branch.txt"
  git -C "${SGLANG_DIR}" rev-parse HEAD > "${OUT_DIR}/sglang_sha.txt"
  git -C "${SGLANG_DIR}" status --short > "${OUT_DIR}/sglang_status.txt"
fi

echo "wrote ${OUT_DIR}"
