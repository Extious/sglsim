from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .wct_cdf_client import POLICY_LABELS


POLICY_STYLE = {
    "aegiskv": ("AegisKV", "#0072B2", "-"),
    "l1_only": ("L1 only", "#009E73", "--"),
    "l2_only": ("L2 only", "#E69F00", "-."),
    "l3_only": ("L3 only", "#CC79A7", (0, (3, 1, 1, 1))),
    "no_backup": ("No backup", "#6B7280", ":"),
}


def load_suite_config(results_dir: Path) -> dict[str, Any]:
    path = results_dir / "suite_config.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def recompute_penalty_label(results_dir: Path) -> str:
    suite = load_suite_config(results_dir)
    if "recompute_ms_per_token" not in suite:
        return ""
    penalty = float(suite["recompute_ms_per_token"])
    multiplier = float(suite.get("recompute_penalty_multiplier", 1.0))
    suffix = f"recompute={penalty:g} ms/token"
    if abs(multiplier - 1.0) > 1e-9:
        suffix += f" ({multiplier:g}x sensitivity)"
    return suffix


def ecdf(values: list[float]) -> tuple[list[float], list[float]]:
    ordered = sorted(values)
    if not ordered:
        return [], []
    return ordered, [(index + 1) / len(ordered) for index in range(len(ordered))]


def load_policy_results(results_dir: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for policy in POLICY_STYLE:
        path = results_dir / f"{policy}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing {policy} result: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("policy") != policy:
            raise ValueError(f"policy mismatch in {path}: {payload.get('policy')}")
        results[policy] = payload
    return results


def write_samples_csv(results: dict[str, dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("policy", "policy_label", "workflow_id", "wct_s"),
        )
        writer.writeheader()
        for policy, payload in results.items():
            for workflow in payload["workflows"]:
                writer.writerow(
                    {
                        "policy": policy,
                        "policy_label": POLICY_LABELS[policy],
                        "workflow_id": workflow["workflow_id"],
                        "wct_s": workflow["wct_s"],
                    }
                )


def render(results_dir: Path, output: Path) -> Path:
    results = load_policy_results(results_dir)
    workflow_sets = {
        policy: {str(row["workflow_id"]) for row in payload["workflows"]}
        for policy, payload in results.items()
    }
    reference = workflow_sets["aegiskv"]
    mismatched = [policy for policy, values in workflow_sets.items() if values != reference]
    if mismatched:
        raise ValueError(f"policies do not contain the same workflow IDs: {mismatched}")
    comparable_fields = (
        "evidence_tier",
        "model",
        "dp_size",
        "trace",
        "workflow_filter",
        "workflow_concurrency",
        "failure_count_requested",
        "failure_manifest",
        "replay",
    )
    reference_payload = results["aegiskv"]
    for policy, payload in results.items():
        differing = [
            field
            for field in comparable_fields
            if payload.get(field) != reference_payload.get(field)
        ]
        if differing:
            raise ValueError(
                f"{policy} is not comparable with aegiskv; differing fields: "
                f"{differing}"
            )

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for policy, (label, color, linestyle) in POLICY_STYLE.items():
        values = [float(row["wct_s"]) for row in results[policy]["workflows"]]
        xs, ys = ecdf(values)
        ax.step(
            xs,
            ys,
            where="post",
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.2,
        )

    sample_count = len(reference)
    failure_count = int(results["aegiskv"].get("failure_count_requested", 0))
    failure_scope = str(
        dict(results["aegiskv"].get("failure_manifest", {}))
        .get("deployment", {})
        .get("failure_scope", "failure")
    )
    replay = results["aegiskv"].get("replay", {})
    input_scale = float(replay.get("input_scale", 1.0))
    context_length = int(replay.get("context_length", 131072))
    profile = (
        f"full-scale proportional replay (<= {context_length:,} tokens)"
        if results["aegiskv"].get("evidence_tier") == "sglsim-online-measured"
        else f"scaled-token smoke ({input_scale:g}x input)"
    )
    if results["aegiskv"].get("failure_manifest"):
        profile += " · fixed time/rank manifest"
    penalty_label = recompute_penalty_label(results_dir)
    if penalty_label:
        profile += f" · {penalty_label}"
    ax.set_xlabel("Workflow completion time (s)")
    ax.set_ylabel("CDF")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(
        f"Real CC trace · {results['aegiskv'].get('model', 'unknown')} · "
        f"context <= {context_length:,} · "
        f"DP={results['aegiskv'].get('dp_size', '?')}/"
        f"PP={results['aegiskv'].get('pp_size', '?')} · "
        f"{sample_count} workflows · {failure_count} {failure_scope} failures\n"
        f"{profile}",
        fontsize=11,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    write_samples_csv(results, output.with_name("wct_samples.csv"))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot five-policy WCT CDF curves.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or (args.results_dir / "wct_cdf.png")
    render(args.results_dir, output)
    print(f"wrote {output}")
    print(f"wrote {output.with_suffix('.pdf')}")
    print(f"wrote {output.with_name('wct_samples.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
