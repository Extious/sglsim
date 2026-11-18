# Algorithm 1: fixed-DAG CPM Slack envelope

## Contract

Agent duration is not Agent Slack.  Real CC is modeled as a fixed staged DAG:
requests of one Agent are sequential, Agents in one stage run in parallel, and
stage boundaries are barriers.  Algorithm 1 first predicts every branch's
duration interval and then applies the CPM max/min transform.

For branch `i` in a fan-out stage:

`Slack_LCB(i) = max(max_j!=i Duration_LCB(j) - Duration_UCB(i) - margin, 0)`

Singleton and critical-path stages have exactly zero Slack.  Algorithm 2 uses
only `Slack_LCB`; median/upper Slack are diagnostics.  Slack is a priority hint,
never a durability certificate: recovery still requires a surviving committed
payload, manifest, and executable path.

## Duration model

The Qwen3-32B request model has three non-negative runtime-fitted terms:

`duration = setup * request_count + prefill_cost * new_prefix_tokens + decode_cost * output_tokens`

Stable Agent prefixes are charged only for growth beyond the largest prior
prompt.  Lower/upper duration scales and the one-sided Slack margin are fitted
from online DP4/PP4 observations, not the old `actual_remaining_s` CSV target.

## Causal online progress correction

The initial CPM pass cannot see which DP rank will execute a branch or how
continuous batching with other workflows will change its effective rate.  The
client therefore re-evaluates criticality from completion prefixes.  For every
branch with completed modeled work `C_i`, total modeled work `W_i`, and latest
observed elapsed time `t_i`, it projects:

`Finish_i = t_i * W_i / C_i`

The online window replaces the static critical ranking only when the projected
leader has completed at least 75% of its requests and still has a future
request to protect.  The decision is made before that next request is sent, so
it can change Algorithm 2 placement.  It consumes no future completion or
barrier time.  This late correction captures routing and batching interference
without changing SGLang dispatch or admission semantics.  The same projected
finish vector updates q50 Slack diagnostics; the conformal Slack LCB is not
made more aggressive by the online point estimate.

## Leakage-free runtime gate

`calibrate_alg1_slack.py` reads the stage/Agent rows emitted by the online WCT
client.  A fixed salt partitions workflow IDs into cost-model fit, conformal
calibration, and final evaluation sets.  Repeated policies or runs of one
workflow cannot cross partitions.

For results collected before the online fields were persisted, the calibrator
causally replays the same progress tracker from measured request-completion
prefixes.  The final stage-start timestamp is used only to reconstruct the
elapsed clock that the live client knew directly; classification never reads a
future request or the barrier outcome.

Promotion requires all of the following on the formal Qwen3-32B, DP=4, PP=4,
32K profile:

- fan-out lower-bound coverage at least 90%;
- critical-branch recall at least 90%;
- critical-branch precision at least 75%;
- p90 absolute Slack error within `max(250 ms, 20% of p90 stage span)`;
- at least 6/3/3 disjoint workflows in fit/calibration/evaluation.

The live artifact is currently explicitly marked `preflight_only`.  It is
accepted only so the new runtime rows can be collected and must not be cited as
final Algorithm-1 accuracy evidence.  The calibrator writes a candidate by
default and replaces the live artifact only with `--promote` after the formal
gate passes.

## Persisted evidence

Every stage/Agent row records actual and predicted duration, actual and
predicted Slack, branch token-work features, interval coverage, and critical
classification, including whether the late progress window replaced the static
ranking and its trigger time.  Every request records calibration ID, semantics,
coverage target, duration interval, Slack interval, and predicted criticality.
