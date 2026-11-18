#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../" && pwd)"
cd "$ROOT"

RESULTS_ROOT="benchmark/AegisKV/presentation/experiments/sglsim_large/results/agentic_workflows"
RUN_PREFIX="aegiskv-alpha-policy-cdf-dp8-48w-fail8"
SCENARIOS=(
  "aegiskv_alpha_policy_cdf_dp8_48w_fail8_fix_a010"
  "aegiskv_alpha_policy_cdf_dp8_48w_fail8_fix_a030"
  "aegiskv_alpha_policy_cdf_dp8_48w_fail8_fix_a050"
  "aegiskv_alpha_policy_cdf_dp8_48w_fail8_mdp"
)
SUFFIXES=(
  "fix_a010"
  "fix_a030"
  "fix_a050"
  "mdp"
)

find_free_port() {
  local start="${1:-39210}"
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

for idx in "${!SCENARIOS[@]}"; do
  scenario="${SCENARIOS[$idx]}"
  suffix="${SUFFIXES[$idx]}"
  run_id="${RUN_PREFIX}-${suffix}-r1"
  port="$(find_free_port $((39210 + idx * 10)))"
  echo "=== scenario=${scenario} run_id=${run_id} start_port=${port} ==="
  uv run python -m benchmark.agentic_workflows.run \
    --scenario "$scenario" \
    --results-root "$RESULTS_ROOT" \
    --run-id "$run_id" \
    --start-port "$port" \
    --log-level info
done

uv run python benchmark/AegisKV/presentation/figures/scripts/render_aegiskv_alpha_policy_wct_cdf.py
