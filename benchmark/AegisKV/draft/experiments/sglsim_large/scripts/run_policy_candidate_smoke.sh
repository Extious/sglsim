#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../" && pwd)"
EXP_DIR="${ROOT_DIR}/benchmark/AegisKV/draft/experiments/sglsim_large"
CONFIG_DIR="${CONFIG_DIR:-${EXP_DIR}/config_policy_candidate}"
RESULTS_ROOT="${RESULTS_ROOT:-${EXP_DIR}/results/policy_candidate_synthesis}"
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
  --output-csv "${LATEST_RUN}/sglsim_policy_candidate_summary.csv" \
  --label "sglsim_policy_candidate"

RUN_ID="sglsim-policy-candidate-$(basename "${LATEST_RUN}")"
"${PYTHON_BIN}" "${EXP_DIR}/scripts/derive_policy_candidates_from_synthesis.py" \
  --run-dir "${LATEST_RUN}" \
  --run-id "${RUN_ID}"

echo "latest_run=${LATEST_RUN}"
