"""Render audit-oriented diagnostics for a completed GlobalDES P0 run.

These figures visualize the mechanics of the twelve-cell contract gate.  They
are deliberately labelled as modeled, contract-only evidence and must not be
used as a performance comparison.  Every plotted row carries source hashes so
that a point can be traced back to the immutable P0 cell artifacts.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sglsim.srt.simulation.global_des_artifacts import (
    EVIDENCE_CLASS,
    MANIFEST_SCHEMA,
)

from .global_des_config import P0_SCENARIOS, REQUIRED_INVARIANTS
from .run_global_des_p0 import AUDIT_DETAILS_SCHEMA, SUMMARY_SCHEMA


OUTPUT_SCHEMA = "aegiskv.global-des-p0-diagnostics.v1"
CLAIM_SCOPE = "contract_only_modeled_not_performance"
DEFAULT_FIGURE_DIRECTORY = "figures"

_CELL_FILES = frozenset(
    {
        "events.jsonl",
        "rank_steps.jsonl",
        "gpu_hbm_timeseries.jsonl",
        "node_dram_timeseries.jsonl",
        "kv_extents.jsonl",
        "transfers.jsonl",
        "transfer_groups.jsonl",
        "transfer_resources.jsonl",
        "recovery_transfers.jsonl",
        "requests.jsonl",
        "failures.jsonl",
        "audit.json",
    }
)
_SLUGS = (
    "event_order_trace",
    "gpu_hbm_accounts",
    "node_dram_accounts",
    "kv_transfer_gantt",
    "invariant_matrix",
)
_LINEAGE_FIELDS = (
    "evidence_class",
    "claim_scope",
    "suite_id",
    "source_code_hash",
    "source_p0_summary_sha256",
    "contract_scenario",
    "source_manifest_payload_sha256",
    "source_event_content_sha256",
    "source_table",
)


class P0DiagnosticError(RuntimeError):
    """Base class for fail-closed P0 diagnostic generation."""


class P0SourceValidationError(P0DiagnosticError):
    """Raised when the P0 source tree is incomplete or not a passing gate."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_canonical_json(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise P0SourceValidationError(f"cannot read canonical JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise P0SourceValidationError(f"expected a JSON object in {path}")
    if payload != _canonical_bytes(value):
        raise P0SourceValidationError(f"JSON is not canonical: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        payload = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise P0SourceValidationError(f"cannot read JSONL {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            raise P0SourceValidationError(f"blank JSONL row at {path}:{number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise P0SourceValidationError(
                f"invalid JSONL row at {path}:{number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise P0SourceValidationError(
                f"JSONL row is not an object at {path}:{number}"
            )
        if (line + "\n").encode("utf-8") != _canonical_bytes(value):
            raise P0SourceValidationError(
                f"JSONL row is not canonical at {path}:{number}"
            )
        rows.append(value)
    return rows


@dataclass(frozen=True)
class _ValidatedP0:
    root: Path
    summary: dict[str, Any]
    summary_sha256: str
    manifests: dict[str, dict[str, Any]]
    audits: dict[str, dict[str, Any]]

    def lineage(self, scenario: str, source_table: str) -> dict[str, Any]:
        manifest = self.manifests[scenario]
        audit = self.audits[scenario]
        return {
            "evidence_class": EVIDENCE_CLASS,
            "claim_scope": CLAIM_SCOPE,
            "suite_id": self.summary["suite_id"],
            "source_code_hash": self.summary["code_hash"],
            "source_p0_summary_sha256": self.summary_sha256,
            "contract_scenario": scenario,
            "source_manifest_payload_sha256": manifest[
                "canonical_payload_sha256"
            ],
            "source_event_content_sha256": audit["details"][
                "event_content_sha256"
            ],
            "source_table": f"{scenario}/{source_table}",
        }


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise P0SourceValidationError(message)


def _validate_manifest_payload(manifest: Mapping[str, Any], path: Path) -> None:
    claimed = manifest.get("canonical_payload_sha256")
    payload = copy.deepcopy(dict(manifest))
    payload.pop("canonical_payload_sha256", None)
    observed = _sha256_bytes(_canonical_bytes(payload))
    _expect(
        isinstance(claimed, str) and claimed == observed,
        f"manifest canonical payload hash mismatch: {path}",
    )


def _validate_file_inventory(cell: Path, manifest: Mapping[str, Any]) -> None:
    inventory = manifest.get("files")
    _expect(isinstance(inventory, dict), f"manifest files is not an object: {cell}")
    _expect(set(inventory) == _CELL_FILES, f"manifest file set mismatch: {cell}")
    for name in sorted(_CELL_FILES):
        descriptor = inventory[name]
        _expect(
            isinstance(descriptor, dict),
            f"invalid manifest descriptor for {cell / name}",
        )
        source = cell / name
        _expect(source.is_file(), f"missing P0 artifact: {source}")
        _expect(
            descriptor.get("sha256") == _sha256_file(source),
            f"artifact SHA256 mismatch: {source}",
        )
        if name == "audit.json":
            row_count = 1
        else:
            row_count = len(source.read_text(encoding="utf-8").splitlines())
        _expect(
            descriptor.get("row_count") == row_count,
            f"artifact row count mismatch: {source}",
        )


def _validate_p0(p0_dir: Path) -> _ValidatedP0:
    root = p0_dir.resolve()
    _expect(root.is_dir(), f"P0 directory does not exist: {root}")
    summary_path = root / "summary.json"
    summary = _read_canonical_json(summary_path)
    summary_sha256 = _sha256_file(summary_path)

    _expect(summary.get("schema") == SUMMARY_SCHEMA, "unexpected P0 summary schema")
    _expect(summary.get("group") == "P0", "source summary is not P0")
    _expect(summary.get("status") == "PASS", "P0 summary must be PASS")
    _expect(
        summary.get("executed_contract_cells")
        == summary.get("required_contract_cells")
        == 12,
        "P0 summary must contain exactly 12 required and executed cells",
    )
    _expect(summary.get("evidence_class") == EVIDENCE_CLASS, "evidence class drift")
    _expect(
        tuple(summary.get("required_invariants", ())) == REQUIRED_INVARIANTS,
        "P0 invariant list drift",
    )
    claims = summary.get("claims")
    _expect(isinstance(claims, dict), "P0 summary claims missing")
    _expect(claims.get("contract_execution_only") is True, "contract-only claim missing")
    _expect(
        claims.get("controller_bridge_target_rotation_closed_loop") is True,
        "target-rotation controller/bridge contract is missing",
    )
    _expect(claims.get("performance_result") is False, "P0 cannot be performance evidence")
    _expect(
        claims.get("real_hardware_measurement") is False,
        "P0 cannot be real-hardware evidence",
    )

    cells = summary.get("cells")
    _expect(isinstance(cells, list) and len(cells) == 12, "invalid P0 cell summary")
    observed_scenarios = tuple(row.get("contract_scenario") for row in cells)
    _expect(observed_scenarios == P0_SCENARIOS, "P0 scenario order/set drift")

    manifests: dict[str, dict[str, Any]] = {}
    audits: dict[str, dict[str, Any]] = {}
    for cell_summary, scenario in zip(cells, P0_SCENARIOS):
        _expect(cell_summary.get("status") == "PASS", f"P0 cell did not pass: {scenario}")
        _expect(
            cell_summary.get("required_invariants_passed")
            == cell_summary.get("required_invariants_total")
            == len(REQUIRED_INVARIANTS),
            f"P0 invariant count mismatch: {scenario}",
        )
        cell = root / scenario
        _expect(cell.is_dir(), f"missing P0 cell directory: {cell}")
        manifest_path = cell / "manifest.json"
        manifest = _read_canonical_json(manifest_path)
        _expect(manifest.get("schema") == MANIFEST_SCHEMA, f"manifest schema drift: {scenario}")
        _expect(manifest.get("status") == "success", f"cell manifest is not successful: {scenario}")
        _expect(manifest.get("evidence_class") == EVIDENCE_CLASS, f"cell evidence drift: {scenario}")
        _expect(manifest.get("code_hash") == summary.get("code_hash"), f"code hash drift: {scenario}")
        simulation = manifest.get("simulation")
        _expect(isinstance(simulation, dict), f"simulation descriptor missing: {scenario}")
        _expect(simulation.get("clock") == "global_virtual_time", f"not global virtual time: {scenario}")
        _expect(simulation.get("timestamp_unit") == "ns", f"timestamp unit drift: {scenario}")
        _expect(simulation.get("real_hardware_measurement") is False, f"hardware claim drift: {scenario}")
        lineage = manifest.get("lineage")
        _expect(isinstance(lineage, dict), f"lineage missing: {scenario}")
        _expect(lineage.get("group") == "P0", f"lineage group drift: {scenario}")
        _expect(lineage.get("contract_scenario") == scenario, f"lineage scenario drift: {scenario}")
        _expect(
            lineage.get("execution_scope") == "executed_contract_cell_only",
            f"execution scope drift: {scenario}",
        )
        _validate_manifest_payload(manifest, manifest_path)
        _expect(
            cell_summary.get("manifest_payload_sha256")
            == manifest.get("canonical_payload_sha256"),
            f"summary/manifest hash mismatch: {scenario}",
        )
        _validate_file_inventory(cell, manifest)

        audit = _read_canonical_json(cell / "audit.json")
        _expect(audit.get("passed") is True and audit.get("errors") == [], f"audit failed: {scenario}")
        _expect(audit.get("evidence_class") == EVIDENCE_CLASS, f"audit evidence drift: {scenario}")
        _expect(audit.get("code_hash") == summary.get("code_hash"), f"audit code hash drift: {scenario}")
        _expect(audit.get("config_sha256") == manifest.get("config_sha256"), f"config hash drift: {scenario}")
        details = audit.get("details")
        _expect(isinstance(details, dict), f"audit details missing: {scenario}")
        _expect(details.get("schema") == AUDIT_DETAILS_SCHEMA, f"audit details schema drift: {scenario}")
        _expect(details.get("status") == "PASS", f"contract audit failed: {scenario}")
        _expect(details.get("contract_scenario") == scenario, f"audit scenario drift: {scenario}")
        invariant_records = details.get("required_invariants")
        _expect(isinstance(invariant_records, dict), f"invariant records missing: {scenario}")
        _expect(
            len(invariant_records) == len(REQUIRED_INVARIANTS)
            and set(invariant_records) == set(REQUIRED_INVARIANTS),
            f"invariant record set drift: {scenario}",
        )
        _expect(
            all(
                isinstance(record, dict)
                and record.get("passed") is True
                and "evidence" in record
                for record in invariant_records.values()
            ),
            f"one or more contract invariants failed: {scenario}",
        )
        _expect(
            details.get("event_content_sha256")
            == cell_summary.get("event_content_sha256"),
            f"event content hash drift: {scenario}",
        )
        detail_claims = details.get("claims")
        _expect(isinstance(detail_claims, dict), f"cell claims missing: {scenario}")
        _expect(detail_claims.get("contract_execution_only") is True, f"contract claim missing: {scenario}")
        _expect(detail_claims.get("full_scheduler_integration") is False, f"scheduler claim drift: {scenario}")
        _expect(
            detail_claims.get(
                "controller_bridge_target_rotation_closed_loop"
            )
            is (scenario == "target_first_and_node_failure"),
            f"target-rotation integration claim drift: {scenario}",
        )
        _expect(detail_claims.get("performance_result") is False, f"performance claim drift: {scenario}")
        _expect(detail_claims.get("real_hardware_measurement") is False, f"hardware claim drift: {scenario}")
        manifests[scenario] = manifest
        audits[scenario] = audit

    return _ValidatedP0(
        root=root,
        summary=summary,
        summary_sha256=summary_sha256,
        manifests=manifests,
        audits=audits,
    )


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.5,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def _mark_contract_only(fig: plt.Figure) -> None:
    fig.text(
        0.995,
        0.006,
        "CONTRACT-ONLY | MODELED | NOT A PERFORMANCE RESULT",
        ha="right",
        va="bottom",
        fontsize=6,
        color="#6B7280",
        weight="bold",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> int:
    if not rows:
        raise P0DiagnosticError(f"refusing to publish an empty diagnostic table: {path.name}")
    fieldnames = list(rows[0])
    if tuple(fieldnames[: len(_LINEAGE_FIELDS)]) != _LINEAGE_FIELDS:
        raise P0DiagnosticError(f"CSV lineage columns missing or reordered: {path.name}")
    if any(list(row) != fieldnames for row in rows):
        raise P0DiagnosticError(f"inconsistent CSV columns: {path.name}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    return len(rows)


def _save_figure(fig: plt.Figure, directory: Path, slug: str) -> None:
    _mark_contract_only(fig)
    pdf = directory / f"{slug}.pdf"
    png = directory / f"{slug}.png"
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={
            "Title": slug,
            "Author": "SGLSim",
            "Subject": "AegisKV contract-only modeled diagnostic",
            "Creator": "SGLSim GlobalDES P0 diagnostics",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        png,
        bbox_inches="tight",
        dpi=220,
        metadata={
            "Title": slug,
            "Author": "SGLSim",
            "Description": "AegisKV contract-only modeled diagnostic; not performance evidence",
            "Software": "SGLSim GlobalDES P0 diagnostics",
        },
    )
    plt.close(fig)


def _event_dispatch_rows(
    source: _ValidatedP0, scenario: str
) -> list[dict[str, Any]]:
    events = _read_jsonl(source.root / scenario / "events.jsonl")
    schedules = {
        str(row["event_id"]): row
        for row in events
        if row.get("kind") == "event_scheduled"
    }
    output: list[dict[str, Any]] = []
    dispatch_order = 0
    for journal_index, row in enumerate(events):
        if row.get("kind") != "event_dispatched":
            continue
        dispatch_order += 1
        scheduled = schedules.get(str(row.get("event_id")))
        _expect(scheduled is not None, f"dispatch without schedule: {scenario}/{row.get('event_id')}")
        lineage = source.lineage(scenario, "events.jsonl")
        output.append(
            {
                **lineage,
                "journal_index": journal_index,
                "dispatch_order": dispatch_order,
                "timestamp_ns": int(row["timestamp_ns"]),
                "timestamp_ms": f"{int(row['timestamp_ns']) / 1_000_000.0:.9f}",
                "event_id": row["event_id"],
                "event_kind": row["entity"],
                "priority": int(scheduled["after"]["priority"]),
                "sequence": int(scheduled["after"]["sequence"]),
                "dispatch_status": row["after"].get("status", ""),
            }
        )
    _expect(bool(output), f"no dispatched events in representative scenario {scenario}")
    return output


def _event_order_diagnostic(source: _ValidatedP0) -> tuple[list[dict[str, Any]], plt.Figure]:
    scenarios = ("qwen3_32b_tp4_shard", "stable_same_timestamp_tie_hash")
    rows = [row for scenario in scenarios for row in _event_dispatch_rows(source, scenario)]
    colors = {0: "#DC2626", 1: "#2563EB", 2: "#059669", 3: "#7C3AED", 5: "#6B7280"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    for ax, scenario, title in zip(
        axes,
        scenarios,
        ("(a) TP4 transfer dispatch", "(b) Same-timestamp tie witness"),
    ):
        selected = [row for row in rows if row["contract_scenario"] == scenario]
        for row in selected:
            ax.scatter(
                float(row["timestamp_ms"]),
                row["dispatch_order"],
                s=34,
                color=colors.get(int(row["priority"]), "#111827"),
                zorder=3,
            )
            label = str(row["event_kind"]).replace("transfer_", "xfer ")
            if len(label) > 24:
                label = label[:23] + "…"
            ax.annotate(
                f"{label}\np={row['priority']}, seq={row['sequence']}",
                (float(row["timestamp_ms"]), row["dispatch_order"]),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=5.7,
            )
        ax.set_xlabel("Global virtual time (ms)")
        ax.set_ylabel("Authoritative dispatch order")
        ax.set_yticks([row["dispatch_order"] for row in selected])
        ax.invert_yaxis()
        ax.grid(alpha=0.2)
        ax.set_title(title)
        if scenario == "stable_same_timestamp_tie_hash":
            ax.set_xlim(-0.00003, 0.00075)
    fig.suptitle("Global event order/time trace (modeled contract witness)", y=1.02)
    fig.tight_layout()
    return rows, fig


_HBM_ACCOUNTS = (
    "weights",
    "runtime",
    "primary_kv",
    "prefix_kv",
    "aegis_pending",
    "aegis_committed",
)
_HBM_COLORS = ("#9CA3AF", "#D1D5DB", "#2563EB", "#7C3AED", "#F59E0B", "#059669")


def _gpu_key(row: Mapping[str, Any]) -> str:
    return (
        f"{row['node_id']}/gpu{row['local_gpu_id']}/dp{row['dp_rank']}/"
        f"pp{row['pp_rank']}/tp{row['tp_rank']}"
    )


def _gpu_hbm_diagnostic(source: _ValidatedP0) -> tuple[list[dict[str, Any]], plt.Figure]:
    scenarios = (
        "failure_after_commit_source_first",
        "failure_in_pending_window",
        "target_first_and_node_failure",
    )
    output: list[dict[str, Any]] = []
    raw_by_scenario: dict[str, list[dict[str, Any]]] = {}
    for scenario in scenarios:
        raw = _read_jsonl(source.root / scenario / "gpu_hbm_timeseries.jsonl")
        if scenario == "target_first_and_node_failure":
            # The rotation witness registers source, old target, replacement,
            # and retry GPUs.  Plot the two standby owners side by side so the
            # old owner's loss/release cannot be mistaken for relabelling.
            raw = [row for row in raw if int(row["dp_rank"]) in {1, 2}]
        raw_by_scenario[scenario] = raw
        lineage = source.lineage(scenario, "gpu_hbm_timeseries.jsonl")
        for sample_order, row in enumerate(raw):
            output.append(
                {
                    **lineage,
                    "sample_order": sample_order,
                    "timestamp_ns": int(row["timestamp_ns"]),
                    "timestamp_ms": f"{int(row['timestamp_ns']) / 1_000_000.0:.9f}",
                    "physical_gpu": _gpu_key(row),
                    "node_id": row["node_id"],
                    "local_gpu_id": row["local_gpu_id"],
                    "dp_rank": row["dp_rank"],
                    "pp_rank": row["pp_rank"],
                    "tp_rank": row["tp_rank"],
                    "weights_gib": f"{int(row['weights']) / 2**30:.9f}",
                    "runtime_gib": f"{int(row['runtime']) / 2**30:.9f}",
                    "primary_kv_gib": f"{int(row['primary_kv']) / 2**30:.9f}",
                    "prefix_kv_gib": f"{int(row['prefix_kv']) / 2**30:.9f}",
                    "aegis_pending_gib": f"{int(row['aegis_pending']) / 2**30:.9f}",
                    "aegis_committed_gib": f"{int(row['aegis_committed']) / 2**30:.9f}",
                    "used_gib": f"{int(row['used_bytes']) / 2**30:.9f}",
                    "capacity_gib": f"{int(row['capacity_bytes']) / 2**30:.9f}",
                    "healthy": bool(row["healthy"]),
                    "reason": row["reason"],
                    "event_id": row["event_id"],
                }
            )

    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.0), sharey=True)
    for row_index, scenario in enumerate(scenarios):
        raw = raw_by_scenario[scenario]
        gpu_keys = sorted({_gpu_key(row) for row in raw})
        _expect(len(gpu_keys) == 2, f"expected exactly two physical GPUs in {scenario}")
        event_rows = _read_jsonl(source.root / scenario / "events.jsonl")
        final_ns = max(int(row["timestamp_ns"]) for row in event_rows)
        display_end_ns = final_ns + max(1, int(final_ns * 0.08))
        for column_index, gpu_key in enumerate(gpu_keys):
            ax = axes[row_index, column_index]
            samples = [row for row in raw if _gpu_key(row) == gpu_key]
            samples.sort(key=lambda item: int(item["timestamp_ns"]))
            x = [int(row["timestamp_ns"]) / 1_000_000.0 for row in samples]
            if int(samples[-1]["timestamp_ns"]) < display_end_ns:
                x.append(display_end_ns / 1_000_000.0)
                samples = [*samples, samples[-1]]
            values = [
                [int(row[account]) / 2**30 for row in samples]
                for account in _HBM_ACCOUNTS
            ]
            ax.stackplot(
                x,
                *values,
                step="post",
                labels=[name.replace("_", " ") for name in _HBM_ACCOUNTS],
                colors=_HBM_COLORS,
                alpha=0.92,
            )
            for sample in samples:
                if sample["reason"] in {"transfer_reserved", "transfer_committed", "transfer_aborted", "gpu_failed"}:
                    ax.axvline(int(sample["timestamp_ns"]) / 1_000_000.0, color="#111827", linewidth=0.45, alpha=0.5)
            outcome = (
                "commit"
                if row_index == 0
                else "abort"
                if row_index == 1
                else "old owner / replacement"
            )
            ax.set_title(f"{outcome}: {gpu_key}", fontsize=7.4)
            ax.set_xlabel("Global virtual time (ms)")
            ax.grid(axis="y", alpha=0.18)
            if column_index == 0:
                ax.set_ylabel("HBM account (GiB)")
            else:
                detail = ax.twinx()
                for account, color in (
                    ("aegis_pending", "#F59E0B"),
                    ("aegis_committed", "#059669"),
                ):
                    detail.step(
                        x,
                        [int(row[account]) / 2**30 for row in samples],
                        where="post",
                        color=color,
                        linewidth=1.8,
                    )
                aegis_peak = max(
                    max(int(row["aegis_pending"]), int(row["aegis_committed"]))
                    for row in samples
                ) / 2**30
                detail.set_ylim(0, max(0.05, aegis_peak * 1.22))
                detail.set_ylabel("Aegis HBM (GiB)", fontsize=6.5)
                detail.tick_params(axis="y", labelsize=6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(
        "Per-physical-GPU HBM ledger: commit, abort, and target rotation",
        y=1.015,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    return output, fig


_DRAM_ACCOUNTS = ("runtime", "hicache_primary", "aegis_pending", "aegis_committed")
_DRAM_COLORS = ("#D1D5DB", "#2563EB", "#F59E0B", "#059669")


def _node_dram_diagnostic(source: _ValidatedP0) -> tuple[list[dict[str, Any]], plt.Figure]:
    scenario = "target_first_and_node_failure"
    raw = _read_jsonl(source.root / scenario / "node_dram_timeseries.jsonl")
    lineage = source.lineage(scenario, "node_dram_timeseries.jsonl")
    output: list[dict[str, Any]] = []
    for sample_order, row in enumerate(raw):
        output.append(
            {
                **lineage,
                "sample_order": sample_order,
                "timestamp_ns": int(row["timestamp_ns"]),
                "timestamp_ms": f"{int(row['timestamp_ns']) / 1_000_000.0:.9f}",
                "node_id": row["node_id"],
                "runtime_gib": f"{int(row['runtime']) / 2**30:.9f}",
                "hicache_primary_gib": f"{int(row['hicache_primary']) / 2**30:.9f}",
                "aegis_pending_gib": f"{int(row['aegis_pending']) / 2**30:.9f}",
                "aegis_committed_gib": f"{int(row['aegis_committed']) / 2**30:.9f}",
                "used_gib": f"{int(row['used_bytes']) / 2**30:.9f}",
                "capacity_gib": f"{int(row['capacity_bytes']) / 2**30:.9f}",
                "healthy": bool(row["healthy"]),
                "reason": row["reason"],
                "event_id": row["event_id"],
            }
        )

    nodes = sorted({str(row["node_id"]) for row in raw})
    _expect(
        nodes == ["node0", "node1", "node2", "node3"],
        "representative DRAM rotation scenario node set drift",
    )
    event_rows = _read_jsonl(source.root / scenario / "events.jsonl")
    final_ns = max(int(row["timestamp_ns"]) for row in event_rows)
    display_end_ns = final_ns + max(1, int(final_ns * 0.08))
    end_ms = display_end_ns / 1_000_000.0
    fig, axes = plt.subplots(len(nodes), 2, figsize=(7.2, 6.7), sharex=True)
    for index, node in enumerate(nodes):
        samples = [row for row in raw if row["node_id"] == node]
        samples.sort(key=lambda item: int(item["timestamp_ns"]))
        x = [int(row["timestamp_ns"]) / 1_000_000.0 for row in samples]
        draw_samples = samples
        if int(samples[-1]["timestamp_ns"]) < display_end_ns:
            x.append(end_ms)
            draw_samples = [*samples, samples[-1]]
        values = [
            [int(row[account]) / 2**30 for row in draw_samples]
            for account in _DRAM_ACCOUNTS
        ]
        axes[index, 0].stackplot(
            x,
            *values,
            step="post",
            labels=[name.replace("_", " ") for name in _DRAM_ACCOUNTS],
            colors=_DRAM_COLORS,
            alpha=0.92,
        )
        axes[index, 0].set_ylabel(f"{node}\nDRAM (GiB)")
        axes[index, 0].grid(axis="y", alpha=0.18)

        for account, color in zip(("aegis_pending", "aegis_committed"), ("#F59E0B", "#059669")):
            axes[index, 1].step(
                x,
                [int(row[account]) / 2**30 for row in draw_samples],
                where="post",
                label=account.replace("_", " "),
                color=color,
                linewidth=1.8,
            )
        for sample in samples:
            if sample["reason"] in {"transfer_reserved", "transfer_committed", "node_dram_failed"}:
                axes[index, 1].axvline(int(sample["timestamp_ns"]) / 1_000_000.0, color="#111827", linewidth=0.55, alpha=0.55)
        axes[index, 1].set_ylabel("Aegis accounts (GiB)")
        axes[index, 1].grid(alpha=0.18)
    axes[0, 0].set_title("Full per-node account ledger")
    axes[0, 1].set_title("Aegis pending / committed detail")
    axes[-1, 0].set_xlabel("Global virtual time (ms)")
    axes[-1, 1].set_xlabel("Global virtual time (ms)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.97))
    fig.suptitle(
        "Per-node DRAM ledger: source-owned copy survives target rotation",
        y=1.01,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.91))
    return output, fig


def _final_transfer_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    state_rank = {"QUEUED": 0, "IN_FLIGHT": 1, "ACKED": 2, "COMMITTED": 3, "ABORTED": 3}
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        transfer_id = str(row["transfer_id"])
        current = selected.get(transfer_id)
        rank = (int(row.get("timestamp_ns", 0)), state_rank.get(str(row.get("state")), -1))
        if current is None:
            selected[transfer_id] = row
            continue
        current_rank = (
            int(current.get("timestamp_ns", 0)),
            state_rank.get(str(current.get("state")), -1),
        )
        if rank >= current_rank:
            selected[transfer_id] = row
    return [selected[key] for key in sorted(selected)]


def _short_resource_path(resources: Sequence[str]) -> str:
    names = [str(resource).split(":", 1)[0] for resource in resources]
    aliases = {
        "source_d2d_engine": "src-D2D",
        "source_d2h_engine": "src-D2H",
        "source_h2d_engine": "src-H2D",
        "source_pcie": "src-PCIe",
        "source_nic": "src-NIC",
        "nvlink_nvswitch": "NVLink/NVSwitch",
        "cross_node_fabric": "fabric",
        "target_nic": "dst-NIC",
        "target_pcie": "dst-PCIe",
        "target_d2d_engine": "dst-D2D",
        "target_h2d_engine": "dst-H2D",
        "source_dram_read_lane": "DRAM-read",
        "destination_dram_write_lane": "DRAM-write",
    }
    return " > ".join(aliases.get(name, name.replace("_", "-")) for name in names)


def _transfer_diagnostic(source: _ValidatedP0) -> tuple[list[dict[str, Any]], plt.Figure]:
    scenario_labels = (
        ("same_node_peer_hbm_path", "same-node peer HBM"),
        ("cross_node_peer_hbm_path", "cross-node peer HBM"),
        ("source_to_node_dram_path", "source to node DRAM"),
        ("node_dram_to_recovery_gpu_path", "node DRAM to recovery GPU"),
        ("failure_in_pending_window", "pending transfer aborted"),
    )
    output: list[dict[str, Any]] = []
    for scenario, label in scenario_labels:
        raw = _read_jsonl(source.root / scenario / "transfers.jsonl")
        final = _final_transfer_rows(raw)
        _expect(len(final) == 1, f"representative transfer scenario must contain one transfer: {scenario}")
        row = final[0]
        resources = [str(value) for value in row["resources"]]
        lineage = source.lineage(scenario, "transfers.jsonl")
        output.append(
            {
                **lineage,
                "path_label": label,
                "transfer_id": row["transfer_id"],
                "extent_id": row["extent_id"],
                "source_tier": row["source"]["tier"],
                "destination_tier": row["destination"]["tier"],
                "start_ns": int(row["start_ns"]),
                "end_ns": int(row["end_ns"]),
                "duration_ns": int(row["end_ns"]) - int(row["start_ns"]),
                "start_ms": f"{int(row['start_ns']) / 1_000_000.0:.9f}",
                "duration_ms": f"{(int(row['end_ns']) - int(row['start_ns'])) / 1_000_000.0:.9f}",
                "final_state": row["state"],
                "resource_path_short": _short_resource_path(resources),
                "resource_path_exact": " | ".join(resources),
            }
        )

    fig, ax = plt.subplots(figsize=(7.2, 3.65))
    color_by_edge = {
        ("gpu_primary", "gpu_aegis"): "#2563EB",
        ("gpu_primary", "node_dram"): "#F59E0B",
        ("node_dram", "gpu_aegis"): "#059669",
    }
    y = np.arange(len(output))
    for index, row in enumerate(output):
        color = color_by_edge.get((str(row["source_tier"]), str(row["destination_tier"])), "#6B7280")
        bars = ax.barh(
            index,
            float(row["duration_ms"]),
            left=float(row["start_ms"]),
            height=0.52,
            color=color,
            edgecolor="#111827",
            linewidth=0.45,
            hatch="///" if row["final_state"] == "ABORTED" else None,
        )
        end = float(row["start_ms"]) + float(row["duration_ms"])
        ax.annotate(
            str(row["final_state"]),
            (end, index),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=6.3,
            weight="bold",
        )
        bars[0].set_gid(str(row["transfer_id"]))
    labels = [f"{row['path_label']}\n{row['resource_path_short']}" for row in output]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Modeled global virtual time (ms; contract exercise only)")
    ax.set_ylabel("KV transfer and contended resource path")
    ax.grid(axis="x", alpha=0.2)
    ax.set_title("KV transfer Gantt / resource-path witnesses")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return output, fig


_INVARIANT_LABELS = {
    "global_event_order": "event order",
    "deterministic_event_hash": "determinism",
    "per_gpu_hbm_capacity": "GPU HBM cap.",
    "per_node_dram_capacity": "node DRAM cap.",
    "kv_extent_conservation": "KV conservation",
    "legal_state_transition": "legal states",
    "all_shards_ack_before_commit": "all-shard ACK",
    "no_hbm_to_dram_edge": "no HBM→DRAM",
    "independent_hbm_dram_admission": "indep. tiers",
    "non_lockstep_dp_overlap": "non-lockstep DP",
}


def _invariant_diagnostic(source: _ValidatedP0) -> tuple[list[dict[str, Any]], plt.Figure]:
    output: list[dict[str, Any]] = []
    matrix = np.zeros((len(P0_SCENARIOS), len(REQUIRED_INVARIANTS)), dtype=int)
    for row_index, scenario in enumerate(P0_SCENARIOS):
        records = source.audits[scenario]["details"]["required_invariants"]
        lineage = source.lineage(scenario, "audit.json")
        for column_index, invariant in enumerate(REQUIRED_INVARIANTS):
            record = records[invariant]
            passed = bool(record["passed"])
            matrix[row_index, column_index] = int(passed)
            output.append(
                {
                    **lineage,
                    "invariant": invariant,
                    "passed": passed,
                    "evidence_sha256": _sha256_bytes(_canonical_bytes(record["evidence"])),
                }
            )
    _expect(bool(np.all(matrix == 1)), "invariant matrix contains a failing cell")

    fig, ax = plt.subplots(figsize=(8.3, 5.0))
    from matplotlib.colors import ListedColormap

    ax.imshow(matrix, vmin=0, vmax=1, aspect="auto", cmap=ListedColormap(["#FEE2E2", "#86EFAC"]))
    ax.set_xticks(
        np.arange(len(REQUIRED_INVARIANTS)),
        [_INVARIANT_LABELS[name] for name in REQUIRED_INVARIANTS],
        rotation=38,
        ha="right",
    )
    ax.set_yticks(np.arange(len(P0_SCENARIOS)), P0_SCENARIOS)
    ax.set_xticks(np.arange(-0.5, len(REQUIRED_INVARIANTS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(P0_SCENARIOS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row_index in range(len(P0_SCENARIOS)):
        for column_index in range(len(REQUIRED_INVARIANTS)):
            ax.text(column_index, row_index, "PASS", ha="center", va="center", fontsize=5.0, color="#14532D", weight="bold")
    ax.set_xlabel("Frozen P0 invariant")
    ax.set_ylabel("Frozen P0 contract scenario")
    ax.set_title("12 scenarios × 10 invariants: contract pass matrix")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return output, fig


def _output_manifest(
    directory: Path,
    source: _ValidatedP0,
    csv_row_counts: Mapping[str, int],
) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for slug in _SLUGS:
        for extension, media_type in (
            ("csv", "text/csv"),
            ("png", "image/png"),
            ("pdf", "application/pdf"),
        ):
            name = f"{slug}.{extension}"
            path = directory / name
            descriptor: dict[str, Any] = {
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
                "media_type": media_type,
            }
            if extension == "csv":
                descriptor["row_count"] = int(csv_row_counts[slug])
            files[name] = descriptor
    payload: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "status": "success",
        "evidence_class": EVIDENCE_CLASS,
        "claim_scope": CLAIM_SCOPE,
        "source": {
            "group": "P0",
            "suite_id": source.summary["suite_id"],
            "code_hash": source.summary["code_hash"],
            "p0_summary_sha256": source.summary_sha256,
            "summary_schema": source.summary["schema"],
            "status": source.summary["status"],
            "contract_cells": 12,
            "invariants_per_cell": len(REQUIRED_INVARIANTS),
        },
        "diagnostics": {
            slug: {
                "csv": f"{slug}.csv",
                "png": f"{slug}.png",
                "pdf": f"{slug}.pdf",
            }
            for slug in _SLUGS
        },
        "files": files,
        "claims": {
            "contract_execution_only": True,
            "full_scheduler_integration": False,
            "controller_bridge_target_rotation_closed_loop": True,
            "performance_result": False,
            "real_hardware_measurement": False,
        },
    }
    payload["canonical_payload_sha256"] = _sha256_bytes(_canonical_bytes(payload))
    return payload


def plot_p0(
    p0_dir: str | os.PathLike[str],
    *,
    figures_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Validate ``RUN_DIR/P0`` and publish five diagnostic bundles atomically.

    The default output is the fresh directory ``RUN_DIR/P0/figures``.  Neither
    the default nor an explicit target is ever overwritten.
    """

    p0_path = Path(p0_dir)
    target = p0_path / DEFAULT_FIGURE_DIRECTORY if figures_dir is None else Path(figures_dir)
    if os.path.lexists(os.fspath(target)):
        raise FileExistsError(f"diagnostic figure directory already exists: {target}")

    source = _validate_p0(p0_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix=f".{target.name}.tmp-", dir=os.fspath(target.parent))
    temp_path = Path(temporary.name)
    _style()
    try:
        builders = (
            ("event_order_trace", _event_order_diagnostic),
            ("gpu_hbm_accounts", _gpu_hbm_diagnostic),
            ("node_dram_accounts", _node_dram_diagnostic),
            ("kv_transfer_gantt", _transfer_diagnostic),
            ("invariant_matrix", _invariant_diagnostic),
        )
        row_counts: dict[str, int] = {}
        for slug, builder in builders:
            rows, figure = builder(source)
            row_counts[slug] = _write_csv(temp_path / f"{slug}.csv", rows)
            _save_figure(figure, temp_path, slug)
        manifest = _output_manifest(temp_path, source, row_counts)
        with (temp_path / "manifest.json").open("xb") as handle:
            handle.write(_canonical_bytes(manifest))
            handle.flush()
            os.fsync(handle.fileno())
        if os.path.lexists(os.fspath(target)):
            raise FileExistsError(f"diagnostic figure directory appeared during publication: {target}")
        os.rename(temp_path, target)
    finally:
        plt.close("all")
        temporary.cleanup()
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render contract-only diagnostics from a passing GlobalDES P0 run"
    )
    parser.add_argument(
        "--p0-dir",
        required=True,
        type=Path,
        help="input RUN_DIR/P0 directory produced by run_global_des_p0.py",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="fresh output directory (default: RUN_DIR/P0/figures)",
    )
    args = parser.parse_args(argv)
    try:
        manifest = plot_p0(args.p0_dir, figures_dir=args.figures_dir)
    except Exception as exc:
        print(f"P0 diagnostic rendering failed: {exc}", file=os.sys.stderr)
        return 2
    output = args.p0_dir / DEFAULT_FIGURE_DIRECTORY if args.figures_dir is None else args.figures_dir
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "claim_scope": manifest["claim_scope"],
                "figures": str(output),
                "manifest": str(output / "manifest.json"),
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "CLAIM_SCOPE",
    "DEFAULT_FIGURE_DIRECTORY",
    "OUTPUT_SCHEMA",
    "P0DiagnosticError",
    "P0SourceValidationError",
    "main",
    "plot_p0",
]
