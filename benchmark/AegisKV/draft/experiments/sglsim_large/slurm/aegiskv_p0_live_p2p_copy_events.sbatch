#!/usr/bin/env bash
#SBATCH --job-name=aegiskv-p0-p2p
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --exclude=hkbugpusrv07
#SBATCH --time=00:20:00
#SBATCH --output=results/strict_final_readiness/slurm-%x-%j.out
#SBATCH --error=results/strict_final_readiness/slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
READINESS_DIR="${ROOT_DIR}/results/strict_final_readiness"
REQUIRED_CSV="${REQUIRED_CSV:-${READINESS_DIR}/p0_physical_tensor_copy_events.required.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-${READINESS_DIR}/p0_physical_tensor_copy_events.p2p_live.csv}"
OUTPUT_JSON="${OUTPUT_JSON:-${READINESS_DIR}/p0_physical_tensor_copy_events.p2p_live.json}"
OUTPUT_MD="${OUTPUT_MD:-${READINESS_DIR}/p0_physical_tensor_copy_events.p2p_live.md}"
BYTES_PER_TOKEN="${BYTES_PER_TOKEN:-147456}"
REPEATS="${REPEATS:-3}"
WARMUP="${WARMUP:-1}"
PEER_SOURCE_DEVICE="${PEER_SOURCE_DEVICE:-0}"
PEER_TARGET_DEVICE="${PEER_TARGET_DEVICE:-1}"

mkdir -p "${READINESS_DIR}"

{
  echo "timestamp=$(date -Iseconds)"
  echo "host=$(hostname)"
  echo "user=$(id -un)"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "slurm_partition=${SLURM_JOB_PARTITION:-}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "root_dir=${ROOT_DIR}"
  echo "python_bin=${PYTHON_BIN}"
  echo "required_csv=${REQUIRED_CSV}"
  echo "output_csv=${OUTPUT_CSV}"
  echo "bytes_per_token=${BYTES_PER_TOKEN}"
  echo "repeats=${REPEATS}"
  echo "warmup=${WARMUP}"
  echo "peer_source_device=${PEER_SOURCE_DEVICE}"
  echo "peer_target_device=${PEER_TARGET_DEVICE}"
  if [ -d "${ROOT_DIR}/.git" ]; then
    git -C "${ROOT_DIR}" branch --show-current || true
    git -C "${ROOT_DIR}" rev-parse HEAD || true
  fi
} | tee "${READINESS_DIR}/p0_live_p2p_collection_provenance_${SLURM_JOB_ID:-manual}.txt"

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L > "${READINESS_DIR}/p0_live_p2p_nvidia-smi-L_${SLURM_JOB_ID:-manual}.txt" || true
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi topo -m > "${READINESS_DIR}/p0_live_p2p_nvidia-smi-topo_${SLURM_JOB_ID:-manual}.txt" || true
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi > "${READINESS_DIR}/p0_live_p2p_nvidia-smi_${SLURM_JOB_ID:-manual}.txt" || true

export PYTHONUNBUFFERED=1

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/collect_p0_live_copy_events.py" \
  --required-csv "${REQUIRED_CSV}" \
  --output-csv "${OUTPUT_CSV}" \
  --output-json "${OUTPUT_JSON}" \
  --output-md "${OUTPUT_MD}" \
  --event-kind live_peer_p2p_copy \
  --bytes-per-token "${BYTES_PER_TOKEN}" \
  --repeats "${REPEATS}" \
  --warmup "${WARMUP}" \
  --peer-source-device "${PEER_SOURCE_DEVICE}" \
  --peer-target-device "${PEER_TARGET_DEVICE}" \
  --strict

echo "wrote ${OUTPUT_CSV}"
