#!/usr/bin/env python3
"""Render an explicitly analytical (not measured) AegisKV scaling figure."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


PRESENTATION_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = PRESENTATION_ROOT / "Figures" / "fig_failure_scaling_theory.png"
CURVE_CSV = (
    PRESENTATION_ROOT
    / "figures"
    / "data"
    / "analytic"
    / "fig_failure_scaling_theory.csv"
)
ASSUMPTIONS_CSV = CURVE_CSV.with_name(
    "fig_failure_scaling_theory_assumptions.csv"
)
NOTE = CURVE_CSV.with_name("fig_failure_scaling_theory.md")


@dataclass(frozen=True)
class CostModel:
    recompute_setup_ms: float = 4.0
    recompute_ms_per_token: float = 0.0108
    l1_setup_ms: float = 2.4
    l1_ms_per_token: float = 0.0019
    l2_setup_ms: float = 5.0
    l2_ms_per_token: float = 0.0036


MODEL = CostModel()
HIGH_RECOMPUTE_MS_PER_TOKEN = 0.22
HORIZON_DAYS = 1.0
ACTIVE_REQUESTS_PER_RANK = 8
FOCUS_TOKENS = 131_072
L1_COVERAGE = 0.25
L2_COVERAGE = 0.50
QWEN3_8B_BYTES_PER_TOKEN = 2 * 36 * 8 * 128 * 2
RETRY_OVERHEAD_MS = MODEL.l1_setup_ms
RECOMPUTE_EXTRA_SETUP_MS = MODEL.recompute_setup_ms - RETRY_OVERHEAD_MS
H2D_EXTRA_SETUP_MS = MODEL.l2_setup_ms - RETRY_OVERHEAD_MS


def probability_at_least_one_failure(
    ranks: np.ndarray,
    *,
    rank_mtbf_days: float,
    horizon_days: float = HORIZON_DAYS,
) -> np.ndarray:
    return 1.0 - np.exp(-ranks * horizon_days / rank_mtbf_days)


def expected_failures(
    ranks: np.ndarray,
    *,
    rank_mtbf_days: float,
    horizon_days: float = HORIZON_DAYS,
) -> np.ndarray:
    return ranks * horizon_days / rank_mtbf_days


def recovery_cost_ms(
    tokens: np.ndarray | float,
    *,
    l1_coverage: float,
    l2_coverage: float,
    recompute_ms_per_token: float = MODEL.recompute_ms_per_token,
) -> np.ndarray:
    tokens_array = np.asarray(tokens, dtype=float)
    phi1 = float(np.clip(l1_coverage, 0.0, 1.0))
    phi2 = float(np.clip(l2_coverage, 0.0, 1.0 - phi1))
    recompute = max(1.0 - phi1 - phi2, 0.0)
    setup = (
        (MODEL.l1_setup_ms if phi1 > 0.0 else 0.0)
        + (MODEL.l2_setup_ms if phi2 > 0.0 else 0.0)
        + (MODEL.recompute_setup_ms if recompute > 0.0 else 0.0)
    )
    return (
        setup
        + tokens_array * phi1 * MODEL.l1_ms_per_token
        + tokens_array * phi2 * MODEL.l2_ms_per_token
        + tokens_array * recompute * recompute_ms_per_token
    )


def full_path_recovery_cost_ms(
    tokens: np.ndarray | float,
    *,
    path: str,
) -> np.ndarray:
    """Idealized full-path costs used by panel (b).

    L1 assumes the complete retry KV is already resident on the target GPU,
    so only fixed retry/setup overhead remains.  L2 assumes the complete KV is
    in Host memory and adds H2D transfer.  No Backup recomputes every token.
    """
    token_array = np.asarray(tokens, dtype=float)
    if path == "no_backup":
        return (
            RETRY_OVERHEAD_MS
            + RECOMPUTE_EXTRA_SETUP_MS
            + token_array * HIGH_RECOMPUTE_MS_PER_TOKEN
        )
    if path == "l1_only":
        return np.full_like(token_array, RETRY_OVERHEAD_MS, dtype=float)
    if path == "l2_only":
        return (
            RETRY_OVERHEAD_MS
            + H2D_EXTRA_SETUP_MS
            + token_array * MODEL.l2_ms_per_token
        )
    raise ValueError(f"unknown recovery path: {path}")


def _write_csv(path: Path, fieldnames: Iterable[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def build_curve_rows() -> list[dict]:
    rows: list[dict] = []
    rank_grid = np.asarray([8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    for mtbf in (30.0, 90.0, 365.0):
        probabilities = probability_at_least_one_failure(
            rank_grid,
            rank_mtbf_days=mtbf,
        )
        for ranks, probability in zip(rank_grid, probabilities):
            rows.append(
                {
                    "panel": "failure_exposure",
                    "series": f"rank_mtbf_{mtbf:g}d",
                    "x_name": "dp_ranks",
                    "x_value": int(ranks),
                    "y_name": "probability_at_least_one_failure_24h",
                    "y_value": float(probability),
                    "l1_coverage": "",
                    "l2_coverage": "",
                    "recompute_ms_per_token": "",
                }
            )

    token_grid = np.asarray([1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
    policies = {
        "no_backup_all_recompute": ("no_backup", 0.0, 0.0),
        "l1_only_full_gpu_reuse": ("l1_only", 1.0, 0.0),
        "l2_only_full_host_reuse": ("l2_only", 0.0, 1.0),
    }
    for series, (path, phi1, phi2) in policies.items():
        costs = full_path_recovery_cost_ms(token_grid, path=path)
        for tokens, cost in zip(token_grid, costs):
            rows.append(
                {
                    "panel": "single_failure_cost",
                    "series": series,
                    "x_name": "context_tokens",
                    "x_value": int(tokens),
                    "y_name": "recovery_cost_ms_per_request",
                    "y_value": float(cost),
                    "l1_coverage": phi1,
                    "l2_coverage": phi2,
                    "recompute_ms_per_token": (
                        HIGH_RECOMPUTE_MS_PER_TOKEN
                        if path == "no_backup"
                        else 0.0
                    ),
                }
            )

    for series, (phi1, phi2, recompute_cost) in {
        "no_backup_default": (0.0, 0.0, MODEL.recompute_ms_per_token),
        "aegiskv_default": (
            L1_COVERAGE,
            L2_COVERAGE,
            MODEL.recompute_ms_per_token,
        ),
        "no_backup_high_recompute": (0.0, 0.0, HIGH_RECOMPUTE_MS_PER_TOKEN),
        "aegiskv_high_recompute": (
            L1_COVERAGE,
            L2_COVERAGE,
            HIGH_RECOMPUTE_MS_PER_TOKEN,
        ),
    }.items():
        per_request_ms = float(
            recovery_cost_ms(
                FOCUS_TOKENS,
                l1_coverage=phi1,
                l2_coverage=phi2,
                recompute_ms_per_token=recompute_cost,
            )
        )
        daily_seconds = (
            expected_failures(rank_grid, rank_mtbf_days=90.0)
            * ACTIVE_REQUESTS_PER_RANK
            * per_request_ms
            / 1000.0
        )
        for ranks, cost in zip(rank_grid, daily_seconds):
            rows.append(
                {
                    "panel": "expected_daily_recovery_work",
                    "series": series,
                    "x_name": "dp_ranks",
                    "x_value": int(ranks),
                    "y_name": "expected_recovery_work_seconds_per_day",
                    "y_value": float(cost),
                    "l1_coverage": phi1,
                    "l2_coverage": phi2,
                    "recompute_ms_per_token": recompute_cost,
                }
            )
    return rows


def write_assumptions() -> None:
    assumptions = [
        ("evidence_type", "analytic_projection", "", "not a measured result"),
        ("failure_process", "independent_poisson", "", "per-rank failures"),
        ("horizon", HORIZON_DAYS, "day", "failure exposure window"),
        ("rank_mtbf_sensitivity", "30;90;365", "day", "not a fitted failure rate"),
        (
            "active_requests_per_failed_rank",
            ACTIVE_REQUESTS_PER_RANK,
            "request/rank",
            "illustrative concurrency",
        ),
        ("focus_context", FOCUS_TOKENS, "token/request", "128K context"),
        (
            "qwen3_8b_kv_bytes_per_token",
            QWEN3_8B_BYTES_PER_TOKEN,
            "byte/token",
            "2 tensors x 36 layers x 8 KV heads x 128 dim x bf16",
        ),
        ("l1_recoverable_coverage", L1_COVERAGE, "fraction", "illustrative, not alpha"),
        ("l2_recoverable_coverage", L2_COVERAGE, "fraction", "illustrative, not beta"),
        ("l1_restore_setup", MODEL.l1_setup_ms, "ms", "SGLSim coefficient model"),
        ("l1_restore_per_token", MODEL.l1_ms_per_token, "ms/token", "SGLSim coefficient model"),
        ("l2_restore_setup", MODEL.l2_setup_ms, "ms", "SGLSim coefficient model"),
        ("l2_restore_per_token", MODEL.l2_ms_per_token, "ms/token", "SGLSim coefficient model"),
        ("recompute_setup", MODEL.recompute_setup_ms, "ms", "SGLSim coefficient model"),
        (
            "recompute_per_token_default",
            MODEL.recompute_ms_per_token,
            "ms/token",
            "conservative SGLSim coefficient model",
        ),
        (
            "recompute_per_token_high",
            HIGH_RECOMPUTE_MS_PER_TOKEN,
            "ms/token",
            "current demo sensitivity setting",
        ),
        (
            "panel_b_retry_overhead",
            RETRY_OVERHEAD_MS,
            "ms/retry",
            "L1 full-GPU-reuse cost; common fixed retry/setup term",
        ),
        (
            "panel_b_h2d_per_token",
            MODEL.l2_ms_per_token,
            "ms/token",
            "L2 full-Host-reuse H2D term",
        ),
        (
            "panel_b_recompute_per_token",
            HIGH_RECOMPUTE_MS_PER_TOKEN,
            "ms/token",
            "No Backup full-recompute term from the current demo profile",
        ),
    ]
    _write_csv(
        ASSUMPTIONS_CSV,
        ("parameter", "value", "unit", "note"),
        [
            {"parameter": key, "value": value, "unit": unit, "note": note}
            for key, value, unit, note in assumptions
        ],
    )


def write_note() -> None:
    NOTE.write_text(
        """# AegisKV analytical failure-scaling model

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
""",
        encoding="utf-8",
    )


def render() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.0))
    rank_grid = np.asarray([8, 16, 32, 64, 128, 256, 512, 1024, 2048])

    ax = axes[0, 0]
    for mtbf, color in ((30.0, "#D55E00"), (90.0, "#E69F00"), (365.0, "#0072B2")):
        probability = 100.0 * probability_at_least_one_failure(
            rank_grid,
            rank_mtbf_days=mtbf,
        )
        ax.plot(rank_grid, probability, marker="o", color=color, label=f"rank MTBF={mtbf:g}d")
    ax.set_xscale("log", base=2)
    ax.set_xticks(rank_grid[::2], [str(value) for value in rank_grid[::2]])
    ax.set_ylim(0, 103)
    ax.set_xlabel("Number of DP ranks")
    ax.set_ylabel("P(at least one failure in 24h) (%)")
    ax.set_title("(a) Failure exposure grows with deployment scale")
    ax.grid(color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    token_grid = np.asarray([1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
    policies = (
        ("No Backup: full recompute", "no_backup", "#6B7280", ":"),
        ("L1 Only: full GPU reuse", "l1_only", "#059669", "-"),
        ("L2 Only: full Host reuse + H2D", "l2_only", "#2563EB", "-."),
    )
    for label, path, color, linestyle in policies:
        cost = full_path_recovery_cost_ms(token_grid, path=path)
        ax.plot(token_grid, cost, marker="o", color=color, linestyle=linestyle, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(token_grid, [f"{value // 1024}K" for value in token_grid])
    ax.set_xlabel("Exposed KV context per request (tokens)")
    ax.set_ylabel("Analytical recovery service time (ms)")
    ax.set_title("(b) Full-path recovery cost from 1K to 128K")
    ax.grid(color="#E5E7EB", linewidth=0.7, which="both")
    ax.legend(frameon=False)
    focus_gib = FOCUS_TOKENS * QWEN3_8B_BYTES_PER_TOKEN / 2**30
    ax.annotate(
        f"128K Qwen3-8B KV ~= {focus_gib:.0f} GiB/request",
        xy=(FOCUS_TOKENS, float(full_path_recovery_cost_ms(FOCUS_TOKENS, path="no_backup"))),
        xytext=(18000, 8000),
        arrowprops={"arrowstyle": "->", "color": "#4B5563"},
        fontsize=8,
    )

    ax = axes[1, 0]
    series = (
        ("No backup, cR=0.0108", 0.0, 0.0, MODEL.recompute_ms_per_token, "#6B7280", ":"),
        ("AegisKV, cR=0.0108", L1_COVERAGE, L2_COVERAGE, MODEL.recompute_ms_per_token, "#059669", "-"),
        ("No backup, cR=0.22", 0.0, 0.0, HIGH_RECOMPUTE_MS_PER_TOKEN, "#6B7280", "--"),
        ("AegisKV, cR=0.22", L1_COVERAGE, L2_COVERAGE, HIGH_RECOMPUTE_MS_PER_TOKEN, "#009E73", "-."),
    )
    for label, phi1, phi2, c_recompute, color, linestyle in series:
        per_request_ms = float(
            recovery_cost_ms(
                FOCUS_TOKENS,
                l1_coverage=phi1,
                l2_coverage=phi2,
                recompute_ms_per_token=c_recompute,
            )
        )
        daily_seconds = (
            expected_failures(rank_grid, rank_mtbf_days=90.0)
            * ACTIVE_REQUESTS_PER_RANK
            * per_request_ms
            / 1000.0
        )
        ax.plot(rank_grid, daily_seconds, marker="o", color=color, linestyle=linestyle, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(rank_grid[::2], [str(value) for value in rank_grid[::2]])
    ax.set_xlabel("Number of DP ranks")
    ax.set_ylabel("Expected recovery work (s/day)")
    ax.set_title("(c) Expected cluster cost (MTBF=90d, 8 active req/rank)")
    ax.grid(color="#E5E7EB", linewidth=0.7, which="both")
    ax.legend(frameon=False, ncol=2)

    ax = axes[1, 1]
    l1_grid = np.linspace(0.0, 1.0, 101)
    l2_grid = np.linspace(0.0, 1.0, 101)
    phi1, phi2 = np.meshgrid(l1_grid, l2_grid)
    valid = phi1 + phi2 <= 1.0 + 1e-12
    remaining = np.maximum(1.0 - phi1 - phi2, 0.0)
    setup = (
        (phi1 > 0.0) * MODEL.l1_setup_ms
        + (phi2 > 0.0) * MODEL.l2_setup_ms
        + (remaining > 0.0) * MODEL.recompute_setup_ms
    )
    aegis_cost = (
        setup
        + FOCUS_TOKENS * phi1 * MODEL.l1_ms_per_token
        + FOCUS_TOKENS * phi2 * MODEL.l2_ms_per_token
        + FOCUS_TOKENS * remaining * MODEL.recompute_ms_per_token
    )
    no_backup_cost = float(recovery_cost_ms(FOCUS_TOKENS, l1_coverage=0.0, l2_coverage=0.0))
    speedup = np.where(valid, no_backup_cost / aegis_cost, np.nan)
    mesh = ax.pcolormesh(phi1, phi2, speedup, shading="auto", cmap="viridis", vmin=1.0, vmax=5.5)
    contours = ax.contour(phi1, phi2, speedup, levels=(1.5, 2.0, 3.0, 4.0), colors="white", linewidths=0.8)
    ax.clabel(contours, inline=True, fontsize=7, fmt="%.1fx")
    ax.scatter([L1_COVERAGE], [L2_COVERAGE], marker="*", s=110, color="#E69F00", edgecolor="black", linewidth=0.5)
    focus_speedup = no_backup_cost / float(
        recovery_cost_ms(
            FOCUS_TOKENS,
            l1_coverage=L1_COVERAGE,
            l2_coverage=L2_COVERAGE,
        )
    )
    ax.annotate(
        f"illustrative point\n{focus_speedup:.2f}x",
        xy=(L1_COVERAGE, L2_COVERAGE),
        xytext=(0.40, 0.68),
        arrowprops={"arrowstyle": "->", "color": "white"},
        color="white",
        fontsize=8,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"Realized L1 recoverable coverage $\phi_1$")
    ax.set_ylabel(r"Realized L2 recoverable coverage $\phi_2$")
    ax.set_title("(d) Theoretical speedup region at 128K context")
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label("No Backup / AegisKV recovery cost")

    fig.suptitle(
        "AegisKV analytical failure-scaling model (projection, not measurement)",
        fontsize=13,
        weight="bold",
    )
    fig.text(
        0.5,
        0.918,
        r"$E[F_H]=NH/M$,  $P(F_H\geq1)=1-e^{-NH/M}$;  "
        r"$C_A=s_1+c_1\phi_1L+s_2+c_2\phi_2L+s_R+c_R(1-\phi_1-\phi_2)L$",
        ha="center",
        fontsize=9,
    )
    fig.text(
        0.5,
        0.012,
        "Independent-failure and service-work model; excludes correlated failures, queueing, detection/model reload, "
        "normal-path backup interference, and DAG slack. Coverage phi is not alpha/beta.",
        ha="center",
        fontsize=8,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.89), h_pad=2.3, w_pad=1.8)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    rows = build_curve_rows()
    _write_csv(
        CURVE_CSV,
        (
            "panel",
            "series",
            "x_name",
            "x_value",
            "y_name",
            "y_value",
            "l1_coverage",
            "l2_coverage",
            "recompute_ms_per_token",
        ),
        rows,
    )
    write_assumptions()
    write_note()
    render()
    print(f"wrote {OUTPUT}")
    print(f"wrote {OUTPUT.with_suffix('.pdf')}")
    print(f"wrote {CURVE_CSV}")
    print(f"wrote {ASSUMPTIONS_CSV}")
    print(f"wrote {NOTE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
