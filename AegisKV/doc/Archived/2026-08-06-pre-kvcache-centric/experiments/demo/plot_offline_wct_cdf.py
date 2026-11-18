"""Plot comparable offline WCT artifacts without retaining their full telemetry.

The logical-time result files contain detailed per-step scheduler telemetry and
can each be hundreds of MiB.  This helper loads one artifact at a time, keeps
only its identity and per-workflow WCT samples, and then renders the ECDF.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .plot_wct_cdf import POLICY_STYLE, ecdf


COMPARABLE_IDENTITY_FIELDS = (
    "model",
    "dp_size",
    "pp_size",
    "context_length",
    "base_server_config",
    "trace",
    "route_plan",
    "failure_manifest",
)


def _parse_result_spec(spec: str) -> tuple[str, Path]:
    policy, separator, raw_path = spec.partition("=")
    if not separator or not policy or not raw_path:
        raise ValueError(f"result must be POLICY=PATH, got {spec!r}")
    if policy not in POLICY_STYLE:
        raise ValueError(f"unsupported policy {policy!r}")
    return policy, Path(raw_path).resolve()


def _load_compact(path: Path, expected_policy: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual_policy = str(payload.get("policy", ""))
    if actual_policy != expected_policy:
        raise ValueError(
            f"policy mismatch in {path}: expected {expected_policy}, "
            f"found {actual_policy}"
        )
    identity = dict(payload.get("identity") or {})
    workflows = {
        str(row["workflow_id"]): float(row["wct_s"])
        for row in payload.get("workflows", [])
    }
    if not workflows:
        raise ValueError(f"no workflow WCT samples in {path}")
    return {
        "path": str(path),
        "policy": expected_policy,
        "identity": {
            field: identity.get(field) for field in COMPARABLE_IDENTITY_FIELDS
        },
        "workflows": workflows,
        "failure_count": int(payload.get("failure_count_injected", 0)),
        "evidence_tier": str(payload.get("evidence_tier", "")),
    }


def render(result_specs: list[str], output: Path) -> Path:
    compact: dict[str, dict[str, Any]] = {}
    for spec in result_specs:
        policy, path = _parse_result_spec(spec)
        if policy in compact:
            raise ValueError(f"duplicate result for policy {policy}")
        compact[policy] = _load_compact(path, policy)

    if "aegiskv" not in compact or "no_backup" not in compact:
        raise ValueError("AegisKV and no_backup results are both required")
    reference = compact["aegiskv"]
    reference_ids = set(reference["workflows"])
    for policy, row in compact.items():
        if row["identity"] != reference["identity"]:
            raise ValueError(f"{policy} identity differs from AegisKV")
        if set(row["workflows"]) != reference_ids:
            raise ValueError(f"{policy} workflow IDs differ from AegisKV")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for policy in POLICY_STYLE:
        if policy not in compact:
            continue
        label, color, linestyle = POLICY_STYLE[policy]
        xs, ys = ecdf(list(compact[policy]["workflows"].values()))
        ax.step(
            xs,
            ys,
            where="post",
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.2,
        )

    identity = reference["identity"]
    ax.set_xlabel("Workflow completion time (s)")
    ax.set_ylabel("Empirical CDF")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(
        "Real agentic trace · "
        f"{identity['model']} · DP={identity['dp_size']}/PP={identity['pp_size']} "
        f"· context={int(identity['context_length']):,}\n"
        f"{len(reference_ids)} paired workflows · "
        f"{reference['failure_count']} matched failures · calibrated SGLSim logical time",
        fontsize=10.5,
    )
    ax.text(
        0.0,
        -0.22,
        "Evidence boundary: online-vs-offline gate passed; not a physical-GPU measurement.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#5E6C84",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    csv_path = output.with_name(f"{output.stem}-samples.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("policy", "workflow_id", "wct_s", "source_artifact"),
        )
        writer.writeheader()
        for policy in POLICY_STYLE:
            if policy not in compact:
                continue
            for workflow_id, wct_s in sorted(compact[policy]["workflows"].items()):
                writer.writerow(
                    {
                        "policy": policy,
                        "workflow_id": workflow_id,
                        "wct_s": wct_s,
                        "source_artifact": compact[policy]["path"],
                    }
                )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot an ECDF from comparable logical-time WCT artifacts."
    )
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        metavar="POLICY=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = render(args.result, args.output)
    print(f"wrote {output}")
    print(f"wrote {output.with_suffix('.pdf')}")
    print(f"wrote {output.with_name(f'{output.stem}-samples.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
