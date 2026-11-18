# AegisKV analytical failure-scaling model

This figure is an analytical projection, not a measured experiment.

For `N` independent DP ranks, a per-rank MTBF of `M` days, and a horizon `H`:

```text
E[F_H] = N H / M
P(F_H >= 1) = 1 - exp(-N H / M)
```

For a request with `L` exposed KV tokens, disjoint L1/L2 recoverable coverage
`phi_1` and `phi_2`, and remaining recomputation coverage
`phi_R = 1 - phi_1 - phi_2`:

```text
C_no(L) = s_R + c_R L
C_A(L)  = I(phi_1>0)s_1 + c_1 phi_1 L
        + I(phi_2>0)s_2 + c_2 phi_2 L
        + I(phi_R>0)s_R + c_R phi_R L
```

Expected aggregate recovery work is `E[F_H] * q * C`, where `q` is the number
of active requests exposed on a failed rank. `phi_1` and `phi_2` are realized
recoverable coverage, not the capacity-control variables alpha and beta.

The model excludes correlated/rack failures, queueing, failure detection,
model reload, normal-path backup interference, and DAG slack. Therefore the
result is recovery-service work, not a direct prediction of workflow WCT.

Panel (b) uses three mutually exclusive full-recovery paths rather than partial
coverage: `No Backup = retry + full recompute`, `L1 Only = retry overhead`, and
`L2 Only = retry + full H2D`. Its recompute coefficient is 0.22 ms/token from
the current demo profile.
