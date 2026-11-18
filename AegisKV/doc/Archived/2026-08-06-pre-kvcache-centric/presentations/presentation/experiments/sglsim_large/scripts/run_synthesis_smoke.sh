#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../" && pwd)"
EXP_DIR="${ROOT_DIR}/benchmark/AegisKV/presentation/experiments/sglsim_large"
CONFIG_DIR="${CONFIG_DIR:-${EXP_DIR}/config_smoke}"
RESULTS_ROOT="${RESULTS_ROOT:-${EXP_DIR}/results/synthesis_smoke}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"

if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="python"
fi

mkdir -p "${RESULTS_ROOT}"

"${PYTHON_BIN}" "${ROOT_DIR}/benchmark/synthesis_requests/run.py" \
  --config-dir "${CONFIG_DIR}" \
  --results-root "${RESULTS_ROOT}" \
  --log-level info \
  "$@"

LATEST_RUN="$(find "${RESULTS_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
if [ -z "${LATEST_RUN}" ]; then
  echo "no run directory found under ${RESULTS_ROOT}" >&2
  exit 1
fi

"${PYTHON_BIN}" "${EXP_DIR}/scripts/collect_synthesis_metrics.py" \
  --run-dir "${LATEST_RUN}" \
  --output-csv "${LATEST_RUN}/sglsim_smoke_summary.csv" \
  --label "sglsim_synthesis_smoke"

echo "latest_run=${LATEST_RUN}"
