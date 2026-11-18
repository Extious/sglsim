"""Write a lineage-local analysis that keeps negative and null results visible."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    24: 2.064,
    29: 2.045,
}


def _f(value: float, digits: int = 1) -> str:
    return f"{float(value):.{digits}f}"


def _t_critical_95(count: int) -> float:
    degrees = max(int(count) - 1, 1)
    if degrees in T_CRITICAL_95:
        return T_CRITICAL_95[degrees]
    smaller = [value for value in T_CRITICAL_95 if value <= degrees]
    return T_CRITICAL_95[max(smaller)] if degrees < 30 and smaller else 1.96


def _mean_ci95(values: pd.Series) -> tuple[float, float, int]:
    clean = values.dropna().astype(float)
    count = int(len(clean))
    mean = float(clean.mean()) if count else float("nan")
    if count <= 1:
        return mean, 0.0, count
    std = float(clean.std(ddof=1))
    return mean, _t_critical_95(count) * std / math.sqrt(count), count


def _group_ci95(
    frame: pd.DataFrame, group_columns: list[str], metric: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouping: Any = group_columns[0] if len(group_columns) == 1 else group_columns
    for keys, group in frame.groupby(grouping, observed=True, sort=True):
        key_values = (keys,) if len(group_columns) == 1 else tuple(keys)
        mean, ci95, count = _mean_ci95(group[metric])
        row = dict(zip(group_columns, key_values))
        row.update({"mean": mean, "ci95": ci95, "seed_count": count})
        rows.append(row)
    return pd.DataFrame(rows)


def _mean_ci_text(mean: float, ci95: float, digits: int = 1) -> str:
    return f"{_f(mean, digits)} ± {_f(ci95, digits)}"


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    rendered = ["| " + " | ".join(headers) + " |"]
    rendered.append("| " + " | ".join("---" for _ in headers) + " |")
    rendered.extend(
        "| " + " | ".join(str(value) for value in row) + " |" for row in rows
    )
    return rendered


def write_analysis_report(lineage_dir: Path, audit: dict[str, Any]) -> Path:
    lineage_dir = Path(lineage_dir)
    manifest = json.loads((lineage_dir / "manifest.json").read_text(encoding="utf-8"))
    provenance = json.loads(
        (lineage_dir / "config_snapshot" / "parameter_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    summary = pd.read_csv(lineage_dir / "summary.csv")
    observations = pd.read_csv(lineage_dir / "observations.csv")
    timeline = pd.read_csv(lineage_dir / "timeline.csv")
    verification = pd.read_csv(lineage_dir / "verification.csv")
    lines = [
        "# AegisKV experiment analysis",
        "",
        f"- Profile: `{manifest['profile']}`; cells: {len(summary)}; "
        f"audit: `{audit['status']}`.",
        f"- Requests simulated across cells: {int(summary.request_count.sum()):,}; "
        f"workflow/session observations: {int(summary.workflow_count.sum()):,}.",
        "- Evidence: complete metadata DES, not physical KV movement or an A100 measurement. "
        "`physical_result=false` and `paper_ready_calibrated=false` remain mandatory.",
        "- S1--S3/S6--S8 use declared load-controlled, time-compressed TraceLab replay; "
        "S4/S5/S9 use controlled mechanism stress; E1 uses recorded AgentX intra-workflow "
        "request timing with a controlled fixed workflow interleave.",
        "- Failure incidence is reported separately from forced conditional injections: the "
        "central 9,600 GPU-hour A100 MTTF is a training-cluster proxy, not an inference-field claim.",
        "- Deployment units are explicit: 200 Gbit/s network line rate and 20 GB/s modeled "
        "effective cross-node payload bandwidth.",
        "- Current-design chunking is machine-checked at 4,096 prefix / 256 decode tokens "
        "for S1--S8 and E1; S9's explicit 1,024-token prefix is a transfer-formula "
        "micro-verification override, not the deployment default.",
        "- Machine-readable provenance is snapshotted and hashed with the lineage. "
        "Paper-readiness blockers: "
        + ", ".join(f"`{name}`" for name in provenance["paper_readiness_blockers"])
        + ".",
        "- `failure_added_wct_p95_ms` is the all-session distribution. S1--S3 instead use "
        "the explicitly conditional harmed-session metric because one forced critical request "
        "cannot change the 95th percentile of dozens of unaffected sessions.",
        "- Uncertainty is the two-sided 95% Student-t confidence interval across independent "
        "deterministic sampling seeds; workflow/session rows are paired within a seed, not "
        "treated as independent replicates.",
        "- Distinct seeds by group: "
        + ", ".join(
            f"{group}={count}"
            for group, count in audit["mechanism_checks"]["distinct_seed_counts"].items()
        )
        + ".",
        "",
    ]

    if "S1" in set(summary.group):
        frame = summary[summary.group == "S1"]
        agg = _group_ci95(
            frame, ["policy"], "affected_failure_added_wct_mean_ms"
        )
        fractions = frame.groupby("policy", as_index=False).agg(
            harmed_fraction=("affected_workflow_fraction", "mean"),
            all_mean=("failure_added_wct_mean_ms", "mean"),
        )
        agg = agg.merge(fractions, on="policy")
        lines.extend(["## S1 Policy comparison", ""])
        lines.extend(
            _table(
                ["Policy", "Harmed-session added WCT (mean ± 95% CI, ms)", "Seeds", "Harmed fraction", "All-session added WCT mean (ms)"],
                [
                    [
                        row.policy,
                        _mean_ci_text(row.mean, row.ci95),
                        int(row.seed_count),
                        f"{100.0 * row.harmed_fraction:.2f}%",
                        _f(row.all_mean),
                    ]
                    for row in agg.itertuples()
                ],
            )
        )
        lines.append("")
        paired = frame.pivot_table(
            index="seed",
            columns="policy",
            values="affected_failure_added_wct_mean_ms",
        )
        paired_delta = paired.aegiskv - paired.recompute
        paired_mean, paired_ci, paired_count = _mean_ci95(paired_delta)
        lines.extend(
            [
                f"Paired effect (AegisKV - recompute): "
                f"`{_mean_ci_text(paired_mean, paired_ci)} ms` "
                f"across {paired_count} seeds.",
                "",
            ]
        )

    if "S2" in set(summary.group):
        frame = summary[summary.group == "S2"]
        agg = _group_ci95(
            frame,
            ["failure_scenario", "load_multiplier", "policy"],
            "affected_failure_added_wct_p95_ms",
        )
        lines.extend(["## S2 Load and fault domain", ""])
        lines.extend(
            _table(
                ["Failure", "Load", "Policy", "Harmed-session p95 added WCT (mean ± 95% CI, ms)"],
                [
                    [row.failure_scenario, _f(row.load_multiplier, 2), row.policy, _mean_ci_text(row.mean, row.ci95)]
                    for row in agg.itertuples()
                ],
            )
        )
        lines.append("")
        paired = frame.pivot_table(
            index=["failure_scenario", "load_multiplier", "seed"],
            columns="policy",
            values="affected_failure_added_wct_p95_ms",
        ).reset_index()
        paired["delta"] = paired.aegiskv - paired.recompute
        paired_agg = _group_ci95(
            paired[paired.failure_scenario != "no_failure"],
            ["failure_scenario", "load_multiplier"],
            "delta",
        )
        lines.extend(["Paired policy effect for injected failures:", ""])
        lines.extend(
            _table(
                ["Failure", "Load", "AegisKV - recompute (mean ± 95% CI, ms)"],
                [
                    [
                        row.failure_scenario,
                        _f(row.load_multiplier, 2),
                        _mean_ci_text(row.mean, row.ci95),
                    ]
                    for row in paired_agg.itertuples()
                ],
            )
        )
        lines.append("")

    if "S3" in set(summary.group):
        frame = summary[summary.group == "S3"]
        agg = _group_ci95(
            frame, ["model", "context_tokens"], "affected_failure_added_wct_p95_ms"
        )
        lines.extend(["## S3 Model and projected context", ""])
        lines.extend(
            _table(
                ["Model", "Projected context", "Harmed-session p95 added WCT (mean ± 95% CI, ms)"],
                [[row.model, int(row.context_tokens), _mean_ci_text(row.mean, row.ci95)] for row in agg.itertuples()],
            )
        )
        lines.append("")

    if "S4" in set(summary.group):
        frame = summary[summary.group == "S4"]
        agg = _group_ci95(frame, ["policy"], "recovery_deadline_miss_max_ms")
        all_p95 = frame.groupby("policy", as_index=False).failure_added_wct_p95_ms.mean()
        agg = agg.merge(all_p95, on="policy")
        lines.extend(["## S4 Algorithm ablation", ""])
        lines.extend(
            _table(
                ["Policy", "Maximum recovery lateness (mean ± 95% CI, ms)", "All-session p95 added WCT mean (ms)"],
                [[row.policy, _mean_ci_text(row.mean, row.ci95), _f(row.failure_added_wct_p95_ms)] for row in agg.itertuples()],
            )
        )
        lines.append("")
        paired = frame.pivot_table(
            index="seed",
            columns="policy",
            values="recovery_deadline_miss_max_ms",
        )
        priority_delta = paired.aegiskv_no_slack - paired.aegiskv
        priority_mean, priority_ci, priority_count = _mean_ci95(priority_delta)
        lines.extend(
            [
                f"Paired no-priority penalty (no causal priority - AegisKV): "
                f"`{_mean_ci_text(priority_mean, priority_ci)} ms` "
                f"across {priority_count} seeds.",
                "",
            ]
        )

    if "S5" in set(summary.group):
        frame = timeline[timeline.group == "S5"]
        agg = frame.groupby("load_multiplier", as_index=False).mean_alpha.agg(["min", "max", "mean"]).reset_index()
        lines.extend(["## S5 HBMQuotaAIMD trajectory", ""])
        lines.extend(
            _table(
                ["Load", "Minimum alpha", "Maximum alpha", "Mean alpha"],
                [[_f(row.load_multiplier, 2), _f(row["min"], 3), _f(row["max"], 3), _f(row["mean"], 3)] for _, row in agg.iterrows()],
            )
        )
        lines.append("")

    if "S6" in set(summary.group):
        frame = observations[observations.group == "S6"].copy()
        frame["reuse_bin"] = pd.cut(
            frame.mean_reuse_fraction,
            [-0.001, 0.90, 0.99, 1.001],
            labels=["<90%", "90--99%", ">=99%"],
        )
        paired = frame.pivot_table(
            index=["seed", "workflow_id", "reuse_bin"],
            columns="policy",
            values="wct_ms",
            observed=True,
        ).reset_index()
        paired["delta"] = paired.aegiskv - paired.recompute
        per_seed = paired.groupby(
            ["seed", "reuse_bin"], observed=True, as_index=False
        ).delta.mean()
        agg = _group_ci95(per_seed, ["reuse_bin"], "delta")
        pair_counts = paired.groupby("reuse_bin", observed=True).size().rename("pair_count").reset_index()
        agg = agg.merge(pair_counts, on="reuse_bin")
        lines.extend(["## S6 Prefix reuse", ""])
        lines.extend(
            _table(
                ["Reuse", "Paired sessions", "Seeds", "AegisKV - recompute (seed mean ± 95% CI, ms)"],
                [[row.reuse_bin, int(row.pair_count), int(row.seed_count), _mean_ci_text(row["mean"], row.ci95)] for _, row in agg.iterrows()],
            )
        )
        lines.append("")

    if "S7" in set(summary.group):
        frame = summary[summary.group == "S7"]
        agg = _group_ci95(
            frame,
            ["repair_time_ms", "effective_bandwidth_GB_per_s"],
            "failure_added_wct_p95_ms",
        )
        lines.extend(["## S7 Bandwidth and repair sensitivity", ""])
        lines.extend(
            _table(
                ["Repair (s)", "Bandwidth (GB/s)", "All-session p95 added WCT (mean ± 95% CI, ms)", "Seeds"],
                [[_f(row.repair_time_ms / 1000.0), _f(row.effective_bandwidth_GB_per_s), _mean_ci_text(row["mean"], row.ci95), int(row.seed_count)] for _, row in agg.iterrows()],
            )
        )
        lines.append("")

    if "S8" in set(summary.group):
        frame = summary[summary.group == "S8"]
        agg = _group_ci95(
            frame, ["failure_scenario", "policy"], "failure_added_wct_p95_ms"
        )
        details = frame.groupby(["failure_scenario", "policy"], as_index=False).agg(
            lateness=("recovery_deadline_miss_max_ms", "mean"),
            failures=("scheduled_failure_count", "mean"),
        )
        agg = agg.merge(details, on=["failure_scenario", "policy"])
        lines.extend(["## S8 Failure count and domain", ""])
        lines.extend(
            _table(
                ["Failure", "Policy", "Failed workers", "All-session p95 added WCT (mean ± 95% CI, ms)", "Maximum recovery lateness mean (ms)"],
                [[row.failure_scenario, row.policy, _f(row.failures, 0), _mean_ci_text(row.mean, row.ci95), _f(row.lateness)] for row in agg.itertuples()],
            )
        )
        lines.append("")
        paired = frame.pivot_table(
            index=["failure_scenario", "seed"],
            columns="policy",
            values="failure_added_wct_p95_ms",
        ).reset_index()
        paired["delta"] = paired.aegiskv - paired.recompute
        paired_agg = _group_ci95(paired, ["failure_scenario"], "delta")
        lines.extend(["Paired end-to-end policy effect:", ""])
        lines.extend(
            _table(
                ["Failure", "AegisKV - recompute (mean ± 95% CI, ms)"],
                [
                    [row.failure_scenario, _mean_ci_text(row.mean, row.ci95)]
                    for row in paired_agg.itertuples()
                ],
            )
        )
        lines.append("")

    if "S9" in set(summary.group):
        frame = verification[verification.group == "S9"]
        lines.extend(
            [
                "## S9 Transfer-model verification",
                "",
                f"The maximum analytical-versus-DES absolute error is `{frame.absolute_error_ms.max():.9f} ms` "
                f"over {len(frame)} exercised recovery transfers. This verifies formula/code agreement, not A100 calibration.",
                "",
            ]
        )

    if "E1" in set(summary.group):
        frame = summary[summary.group == "E1"]
        agg = _group_ci95(
            frame, ["policy"], "affected_failure_added_wct_mean_ms"
        )
        paired = observations[observations.group == "E1"].pivot_table(
            index=["seed", "workflow_id"],
            columns="policy",
            values="failure_added_wct_ms",
        ).reset_index()
        paired["delta"] = paired.aegiskv - paired.recompute
        per_seed = paired.groupby("seed", as_index=False).delta.mean()
        paired_mean, paired_ci, paired_count = _mean_ci95(per_seed.delta)
        lines.extend(["## E1 AgentX/Weka external validity", ""])
        lines.extend(
            _table(
                ["Policy", "Harmed-workflow added WCT (mean ± 95% CI, ms)", "Seeds"],
                [
                    [row.policy, _mean_ci_text(row.mean, row.ci95), int(row.seed_count)]
                    for row in agg.itertuples()
                ],
            )
        )
        lines.extend(
            [
                "",
                f"Paired all-workflow effect (AegisKV - recompute): "
                f"`{_mean_ci_text(paired_mean, paired_ci)} ms` across "
                f"{paired_count} workflow-sampling seeds.",
                "",
                "This external-validity group preserves the public AgentX/Weka "
                "intra-workflow request schedule, observed subagent DAG, and prefix "
                "hashes. Workflow starts and Qwen3 token sizes are controlled; source "
                "API durations are not reused as Qwen service times.",
                "",
            ]
        )

    diagnoses: list[str] = []
    s1 = summary[summary.group == "S1"]
    if not s1.empty and {"aegiskv", "recompute"}.issubset(set(s1.policy)):
        means = s1.groupby("policy").affected_failure_added_wct_mean_ms.mean()
        if means["recompute"] > 0:
            reduction = 100.0 * (means["recompute"] - means["aegiskv"]) / means["recompute"]
            diagnoses.append(
                f"S1's harmed-session mean is {_f(reduction)}% lower for AegisKV than recompute in this modeled deployment."
            )
        protected = s1[s1.policy.isin(["host_checkpoint", "neighbor_replica", "aegiskv"])]
        if protected.groupby("policy").affected_failure_added_wct_mean_ms.mean().nunique() == 1:
            diagnoses.append(
                "Host, neighbor, and AegisKV tie in S1 because the configured committed-copy load paths share the same effective bandwidth; this suite does not invent a tier-speed advantage."
            )
    s4 = summary[summary.group == "S4"]
    if not s4.empty:
        means = s4.groupby("policy").recovery_deadline_miss_max_ms.mean()
        if {"aegiskv", "aegiskv_no_slack"}.issubset(means.index):
            paired = s4.pivot_table(
                index="seed",
                columns="policy",
                values="recovery_deadline_miss_max_ms",
            )
            delta_mean, delta_ci, _count = _mean_ci95(
                paired.aegiskv_no_slack - paired.aegiskv
            )
            diagnoses.append(
                f"Causal priority reduces maximum recovery lateness by a paired "
                f"{_mean_ci_text(delta_mean, delta_ci)} ms in the controlled fan-in test; "
                "total WCT remains nearly unchanged."
            )
    s8 = summary[summary.group == "S8"]
    if not s8.empty:
        pivot = s8.groupby(["failure_scenario", "policy"]).failure_added_wct_p95_ms.mean().unstack()
        wins = int((pivot.aegiskv < pivot.recompute).sum())
        paired = s8.pivot_table(
            index=["failure_scenario", "seed"],
            columns="policy",
            values="failure_added_wct_p95_ms",
        ).reset_index()
        paired["delta"] = paired.aegiskv - paired.recompute
        paired_ci = _group_ci95(paired, ["failure_scenario"], "delta")
        inconclusive = int(
            ((paired_ci["mean"] - paired_ci["ci95"] <= 0.0)
             & (paired_ci["mean"] + paired_ci["ci95"] >= 0.0)).sum()
        )
        diagnoses.append(
            f"AegisKV has lower mean all-session p95 added WCT in {wins}/{len(pivot)} "
            f"S8 failure scenarios, while {inconclusive}/{len(paired_ci)} paired 95% CIs "
            "include zero. Recovery lateness improves more consistently, so the current "
            "DES does not support a blanket end-to-end dominance claim."
        )
    s7 = summary[summary.group == "S7"]
    if not s7.empty:
        grouped = s7.groupby(
            ["repair_time_ms", "effective_bandwidth_GB_per_s"]
        ).failure_added_wct_p95_ms.agg(["mean", "std"])
        median_cv = (grouped["std"] / grouped["mean"].replace(0, float("nan"))).median()
        diagnoses.append(
            f"S7 has a median across-seed coefficient of variation of {_f(100.0 * median_cv)}%; bandwidth/repair trends are identifiable but noisy."
        )
        repair_means = s7.groupby(
            ["repair_time_ms", "effective_bandwidth_GB_per_s"]
        ).failure_added_wct_p95_ms.mean().unstack()
        if {15000, 30000}.issubset(repair_means.index) and repair_means.loc[
            15000
        ].equals(repair_means.loc[30000]):
            diagnoses.append(
                "S7's 15 s and 30 s rows are exactly equal at every bandwidth. "
                "The raw events show recovery route activation before worker repair and "
                "zero degraded-route rebinds, so additional outage duration is non-binding "
                "for this finite trace; the plateau is not a plotting aggregation error."
            )
    e1 = summary[summary.group == "E1"]
    if not e1.empty:
        paired = observations[observations.group == "E1"].pivot_table(
            index=["seed", "workflow_id"],
            columns="policy",
            values="failure_added_wct_ms",
        ).reset_index()
        paired["delta"] = paired.aegiskv - paired.recompute
        per_seed = paired.groupby("seed", as_index=False).delta.mean()
        mean, ci95, _count = _mean_ci95(per_seed.delta)
        improved = int((per_seed.delta < -1e-9).sum())
        tied = int((per_seed.delta.abs() <= 1e-9).sum())
        worsened = int((per_seed.delta > 1e-9).sum())
        if mean - ci95 <= 0.0 <= mean + ci95:
            diagnoses.append(
                f"E1 is statistically inconclusive: its paired all-workflow effect is "
                f"{_mean_ci_text(mean, ci95)} ms, with {improved} improved, {tied} tied, "
                f"and {worsened} worsened seed partitions. Conservative AgentX Slack "
                "LCBs fail close to zero, so any benefit here comes from committed "
                "protection extents rather than recovery-priority evidence."
            )
    if "S5" in set(summary.group):
        diagnoses.append(
            "S5 verifies that alpha changes and the no-AIMD control remains fixed; "
            "it does not by itself prove an end-to-end latency benefit."
        )
    diagnoses.append(
        "The largest remaining evidence blocker is same-topology A100 service/transfer "
        "calibration followed by matched physical replay. Reviewer acceptance cannot be "
        "guaranteed from metadata-DES evidence alone."
    )
    lines.extend(["## Diagnosis against the design theory", ""])
    lines.extend(f"- {item}" for item in diagnoses)
    lines.extend(["", "## Figures", ""])
    for path in sorted((lineage_dir / "figures").glob("*.png")):
        lines.append(f"- [{path.stem}](figures/{path.name})")
        lines.append("")

    output = lineage_dir / "analysis.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


__all__ = ["write_analysis_report"]
