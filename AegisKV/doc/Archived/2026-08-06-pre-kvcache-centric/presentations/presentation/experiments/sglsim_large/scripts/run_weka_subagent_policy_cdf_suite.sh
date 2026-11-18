#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../" && pwd)"
cd "$ROOT"

RESULTS_ROOT="benchmark/AegisKV/presentation/experiments/sglsim_large/results/agentic_workflows"
RUN_PREFIX="weka-subagent-policy-cdf-dp8-24w-fail6"
SCENARIOS=(
  "weka_subagent_policy_cdf_dp8_24w_fail6_no_backup"
  "weka_subagent_policy_cdf_dp8_24w_fail6_aegiskv"
)
SUFFIXES=(
  "no_backup"
  "aegiskv"
)

find_free_port() {
  local start="${1:-39100}"
  local port="$start"
  while [ "$port" -lt $((start + 200)) ]; do
    if ! ss -tln 2>/dev/null | awk '{print $4}' | grep -q ":${port}$"; then
      echo "$port"
      return 0
    fi
    port=$((port + 10))
  done
  echo "no free port found near ${start}" >&2
  return 1
}

uv run python -m benchmark.agentic_workflows.weka_trace --batch-cdf

for idx in "${!SCENARIOS[@]}"; do
  scenario="${SCENARIOS[$idx]}"
  suffix="${SUFFIXES[$idx]}"
  run_id="${RUN_PREFIX}-${suffix}-r1"
  port="$(find_free_port $((39100 + idx * 10)))"
  echo "=== scenario=${scenario} run_id=${run_id} start_port=${port} ==="
  uv run python -m benchmark.agentic_workflows.run \
    --scenario "$scenario" \
    --results-root "$RESULTS_ROOT" \
    --run-id "$run_id" \
    --start-port "$port" \
    --log-level info
done

uv run python benchmark/AegisKV/presentation/figures/scripts/render_weka_subagent_policy_wct_cdf.py
