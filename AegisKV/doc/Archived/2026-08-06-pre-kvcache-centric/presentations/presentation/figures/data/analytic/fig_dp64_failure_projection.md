# DP64 Real CC failure projection

This artifact is an analytical Monte-Carlo projection, not an SGLSim or
physical-system measurement.

The user-supplied availability image is digitized and calibrated to its stated
mean of 58.07 available GPUs over 15 hours.  For a request placed at time `t`,
the projection uses `(64 - A(t)) / 64` as its retry exposure probability.  This
is an exposure approximation; it is not a reconstructed failure event log.

The baseline workflow time is the Real CC trace arrival span.  Only retries on
trace-marked critical agents add visible recovery delay.  Normal-path backup
interference, queueing redistribution, model reload, correlated failure
domains, and exact DAG slack propagation are excluded.

For AegisKV, fixed alpha=0.25 and beta=0.50 are capacity fractions, not direct
reuse probabilities.  The current SGLSim A100 profile provides 176,085 L1 and
352,170 L2 KV tokens per rank.  This revision directly calls the runtime
`AegisRankBackupManager.place()` implementation of Algorithm 2: critical
requests use probability 0.90 and zero slack; non-critical requests use 0.35
and 8 ms slack; 256-token candidates undergo value-density admit/replace.
Entries compete in bounded active sets of at most eight requests per target
rank and are released when an active set closes.

The trace lacks Router queue snapshots.  Execution/backup ranks therefore use
a deterministic balanced DP64 proxy.  To isolate Algorithm 2, a retry is
assumed to consume the ACKed Backup rank; Router load-versus-reuse rerouting is
not modeled here.
