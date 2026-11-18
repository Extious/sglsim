#!/usr/bin/env python3
"""Fail-closed contract for the isolated RTX4090 PP2 Mooncake lineage.

This module deliberately does not share the V100 schema-v2 consumer.  The
RTX4090 producer is a different hardware/layout/transport lineage and its
current registered-GPU-memory transfers report ``transfer_protocol=tcp``.
That fact is preserved as part of the profile instead of being relabelled as
CUDA P2P/D2D.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
for search_root in (REPO_ROOT, REPO_ROOT / "python"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

from sglsim.srt.simulation.aegis_independent_tiers import DESIGN_SEMANTICS


HARDWARE_LABEL = "RTX4090-2node-PP2-systemDP2"
FALLBACK_ID = "qwen3-8b-rtx4090-2node-pp2-system-dp2-v1"
MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
SOURCE_ENTRIES_SHA256 = (
    "9ccd7da9e870c5ee962c235c50e2cfc5ef0763422f3c7f1389c838174c58e0e3"
)
PROFILE_SEMANTICS = "rtx4090_pp2_registered_hbm_tcp_four_path_v1"
PROFILE_EVIDENCE_CLASS = "sglsim_rtx4090_pp2_lineage_profile"
IMPORT_EVIDENCE_CLASS = "sglsim_rtx4090_pp2_lineage_import"
LAYOUT = {"tp_size": 1, "pp_size": 2, "dp_size": 2}
NODE_ORDER = ["gpu11", "gpu13"]
RUNTIME_CONSUMPTION_CONTRACT = {
    "tier_recovery_ms": {
        "peer_hbm": "peer_hbm_recovery_effective_bytes_per_ms",
        "dram": "dram_recovery_effective_bytes_per_ms",
    },
    "protection_observability": {
        "peer_hbm": "peer_hbm_protection_effective_bytes_per_ms",
        "dram": "direct_dram_protection_effective_bytes_per_ms",
    },
    "protection_rate_is_subtracted_from_admission_gain": False,
    "cross_tier_transfer_cost_is_charged": False,
}
PROFILE_METRICS = (
    "recompute_ms_per_token",
    "recompute_fixed_overhead_ms",
    "peer_hbm_protection_effective_bytes_per_ms",
    "direct_dram_protection_effective_bytes_per_ms",
    "peer_hbm_recovery_effective_bytes_per_ms",
    "dram_recovery_effective_bytes_per_ms",
)
TCP_DIAGNOSTIC_ACK = "registered_hbm_over_tcp_diagnostic_v1"
INVALID_ATTEMPT_NOTE = (
    "No structurally invalid attempt supplied performance values."
)


@dataclass(frozen=True)
class InputPaths:
    topology_gate: Path
    serve_gate: Path
    recompute_gate: Path
    dram_gate: Path
    peer_hbm_gate: Path
    integration_gate: Path
    aimd_gate: Path
    recovery_gate: Path
    series_audit: Path
    figure_audit: Path
    figure_png: Path
    figure_pdf: Path
    figure_csv: Path
    preliminary_report: Path
    full_request_baseline_first: Path
    full_request_aegis_first: Path

    def as_dict(self) -> dict[str, Path]:
        return {field.name: Path(getattr(self, field.name)) for field in fields(self)}


EXPECTED_ARTIFACTS: dict[str, tuple[str, str]] = {
    "topology_gate": (
        "aegiskv/runs/fallback-4090-topology-135388-20260824T054622Z/"
        "topology-gate-report.json",
        "de04291819c704f98223dbd386c01597ddabd47436bb39a0f32899f0043891cd",
    ),
    "serve_gate": (
        "aegiskv/runs/fallback-4090-vllm-pp-135393-20260824T054729Z/"
        "serve-smoke-gate-report.json",
        "d17bf9b8b82ab63e720400d3c60e148897a95919b5947892057fd0804e78a76f",
    ),
    "recompute_gate": (
        "aegiskv/runs/fallback-4090-recompute-135399-20260824T054924Z/"
        "recompute-calibration-gate-report.json",
        "5d90ecbc98cfdd7850882bd91ccaef45181bc8c2ce8ac47e763d3d2678b4b9ec",
    ),
    "dram_gate": (
        "aegiskv/runs/fallback-4090-mooncake-dram-135416-20260824T055408Z/"
        "mooncake-dram-calibration-gate-report.json",
        "06461b479c1753fbf9bdb139afabc33fb903537dc94d712ab66059c218e3fb2b",
    ),
    "peer_hbm_gate": (
        "aegiskv/runs/fallback-4090-peer-hbm-135408-20260824T055148Z/"
        "peer-hbm-calibration-gate-report.json",
        "5fbfa14612858feca93ecf23b840153c80c1def7c6e4e8a48f1fefce085db706",
    ),
    "integration_gate": (
        "aegiskv/runs/fallback-4090-aegis-hbm-135430-20260824T055825Z/"
        "aegis-hbm-integration-gate-report.json",
        "1e20edbb89dba7781fa19442bb0f0278beb0a5df69957e43bad47d4441fe4a83",
    ),
    "aimd_gate": (
        "aegiskv/runs/fallback-4090-aegis-aimd-135440-20260824T060113Z/"
        "aegis-aimd-integration-gate-report.json",
        "bceb2c28648dd67d919b84afd2e7fea448e478dceaf2b5014be6bd74cee1bcd0",
    ),
    "recovery_gate": (
        "aegiskv/runs/fallback-4090-aegis-hbm-recovery-135452-20260824T060425Z/"
        "aegis-hbm-recovery-gate-report.json",
        "d92210155a6704a77725f3c5d35f985cb34dae21ae586470e90a7bef8dbe2d08",
    ),
    "series_audit": (
        "aegiskv/runs/fallback-4090-pp2-paired-series-seeds-20260822-20260825/"
        "paired-series-audit.json",
        "7a1afb53aa2935cee0aa6d7f8756c0fea354661caa8727566cf6e9de68423d22",
    ),
    "figure_audit": (
        "aegiskv/runs/fallback-4090-pp2-paired-series-seeds-20260822-20260825/"
        "figure-v3/paired-series-figure-audit.json",
        "edc50da1ab0bb7edf323155318085ddb78851def6b8e5f4b075c0cfc40480751",
    ),
    "figure_png": (
        "aegiskv/runs/fallback-4090-pp2-paired-series-seeds-20260822-20260825/"
        "figure-v3/paired-series-effect.png",
        "32444be339779ffdf09fb44beaca24360813a046ec594e779de88505a049549c",
    ),
    "figure_pdf": (
        "aegiskv/runs/fallback-4090-pp2-paired-series-seeds-20260822-20260825/"
        "figure-v3/paired-series-effect.pdf",
        "c290af80a6cfe9e99a87a305af5fb211364cf9b3cd4e321f5647c1d1fb3e1345",
    ),
    "figure_csv": (
        "aegiskv/runs/fallback-4090-pp2-paired-series-seeds-20260822-20260825/"
        "figure-v3/paired-series.csv",
        "d66b2dbeda7fa286f8072473153c03670a05a2f82ba636217112d3bceb930182",
    ),
    "preliminary_report": (
        "aegiskv/runs/fallback-4090-pp2-paired-series-seeds-20260822-20260825/"
        "PRELIMINARY_REPORT.md",
        "658a58e855841c61794e50d57b90350d764c32ba91e1e390e37c90ca523d6033",
    ),
    "full_request_baseline_first": (
        "aegiskv/runs/fallback-4090-pp2-paired-pilot-135463-20260824T060741Z/"
        "paired-pilot-report.json",
        "5380571d744d71d99495615ee004f53b5cee687ab6a1b80365a8653a88445ec8",
    ),
    "full_request_aegis_first": (
        "aegiskv/runs/fallback-4090-pp2-paired-pilot-135499-20260824T061752Z/"
        "paired-pilot-report.json",
        "b2249f4894b2ec31f1cf75457239cf87fb77e0494c62ec47c725a9c67f8c64c4",
    ),
}

GATE_CONTRACTS = {
    "topology_gate": ("135388", "aegiskv_4090_fallback_topology_gate"),
    "serve_gate": ("135393", "aegiskv_4090_vllm_pp_serve_smoke_gate"),
    "recompute_gate": (
        "135399",
        "aegiskv_4090_pp_recompute_calibration_gate",
    ),
    "dram_gate": ("135416", "aegiskv_4090_pp_mooncake_dram_calibration_gate"),
    "peer_hbm_gate": ("135408", "aegiskv_4090_pp2_peer_hbm_calibration_gate"),
    "integration_gate": ("135430", "aegiskv_4090_pp2_hbm_integration_gate"),
    "aimd_gate": ("135440", "aegiskv_4090_pp2_stage_aggregated_aimd_gate"),
    "recovery_gate": ("135452", "aegiskv_4090_pp2_hbm_recovery_gate"),
}


def default_input_paths(
    mooncake_root: Path = Path("/home/comp/zhanzhao/Mooncake"),
) -> InputPaths:
    root = mooncake_root.resolve()
    return InputPaths(
        **{
            name: root / relative
            for name, (relative, _sha256) in EXPECTED_ARTIFACTS.items()
        }
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def artifact_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def verify_artifact(record: dict[str, Any], *, label: str) -> Path:
    if not isinstance(record, dict) or not {"path", "sha256"} <= set(record):
        raise ValueError(f"{label} artifact record is incomplete")
    path = Path(str(record["path"])).resolve()
    if not path.is_file():
        raise ValueError(f"{label} artifact is missing: {path}")
    actual = sha256_file(path)
    if actual != str(record["sha256"]):
        raise ValueError(f"{label} artifact sha256 mismatch")
    if "size_bytes" in record and path.stat().st_size != int(record["size_bytes"]):
        raise ValueError(f"{label} artifact size mismatch")
    return path


def verify_embedded_artifact_tree(value: Any, *, label: str) -> None:
    """Re-hash every conventional artifact record below ``value``."""

    if isinstance(value, dict):
        if {"path", "sha256"} <= set(value):
            path = verify_artifact(value, label=label)
            if "value" in value:
                embedded = load_object(path)
                if value["value"] != embedded:
                    raise ValueError(f"{label} embedded JSON mismatch")
                verify_embedded_artifact_tree(
                    embedded, label=f"{label}.embedded"
                )
            return
        for name, child in value.items():
            verify_embedded_artifact_tree(child, label=f"{label}.{name}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            verify_embedded_artifact_tree(child, label=f"{label}[{index}]")


def _expected_artifact(path: Path, *, name: str) -> dict[str, Any]:
    path = path.resolve()
    relative, expected_sha = EXPECTED_ARTIFACTS[name]
    if not path.as_posix().endswith(relative):
        raise ValueError(f"{name} path is not the pinned RTX4090 PP2 artifact")
    record = artifact_record(path)
    if record["sha256"] != expected_sha:
        raise ValueError(f"{name} sha256 is not the pinned RTX4090 PP2 artifact")
    return record


def _positive_number(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return number


def _nonnegative_number(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be nonnegative and finite")
    return number


def _common_gate(value: dict[str, Any], *, name: str) -> None:
    job_id, evidence_class = GATE_CONTRACTS[name]
    if not (
        value.get("schema_version") == 1
        and value.get("status") == "PASS"
        and value.get("failures") == []
        and value.get("design_semantics") == DESIGN_SEMANTICS
        and value.get("evidence_class") == evidence_class
        and str(value.get("slurm_job_id")) == job_id
        and value.get("result_label") == HARDWARE_LABEL
        and value.get("paper_ready") is False
        and value.get("node_order") == NODE_ORDER
        and value.get("pilot_ready", False) is False
        and value.get("formal_ready", False) is False
        and value.get("formal_pilot_ready", False) is False
    ):
        raise ValueError(f"{name} contract did not pass")


def _verify_source_manifest(
    record: dict[str, Any], *, label: str, expected_entries: list[dict[str, Any]] | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path_text = str(record.get("path") or "")
    if not path_text:
        raise ValueError(f"{label} source manifest path is missing")
    path = Path(path_text).resolve()
    if "sha256" in record:
        verify_artifact(record, label=f"{label} source manifest")
    elif not path.is_file():
        raise ValueError(f"{label} source manifest is missing")
    manifest = load_object(path)
    entries = manifest.get("entries")
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("evidence_class")
        == "aegiskv_4090_fallback_source_config_manifest"
        and manifest.get("status") == "PINNED"
        and manifest.get("design_semantics") == DESIGN_SEMANTICS
        and manifest.get("fallback_id") == FALLBACK_ID
        and manifest.get("node_order") == NODE_ORDER
        and manifest.get("separate_from_v100_campaign") is True
        and manifest.get("v100_source_manifest_reused") is False
        and isinstance(entries, list)
        and len(entries) == 175
        and int(manifest.get("entry_count", -1)) == len(entries)
        and manifest.get("entries_sha256") == SOURCE_ENTRIES_SHA256
        and canonical_sha256(entries) == SOURCE_ENTRIES_SHA256
    ):
        raise ValueError(f"{label} source manifest contract did not pass")
    if record.get("entries_sha256") != SOURCE_ENTRIES_SHA256:
        raise ValueError(f"{label} source-manifest record digest mismatch")
    if expected_entries is not None and entries != expected_entries:
        raise ValueError(f"{label} source entries differ from the frozen lineage")
    return manifest, entries


def _validate_frozen_source_entries(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate the immutable manifest without depending on the mutable checkout.

    The producer was hardened after the frozen 4090 run, so its live checkout is
    expected to differ from these 175 historical entries.  Requiring current
    files to match would make an immutable run impossible to import after any
    legitimate producer fix.  Every gate's pinned source-manifest artifact and
    its canonical entry digest are still re-hashed above.
    """

    seen: set[str] = set()
    for index, entry in enumerate(manifest["entries"]):
        relative = str(entry.get("relative_path") or "")
        relative_path = Path(relative)
        digest = str(entry.get("sha256") or "")
        if (
            not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative in seen
            or len(digest) != 64
            or int(entry.get("size_bytes", -1)) < 0
        ):
            raise ValueError(f"unsafe frozen source entry {index}")
        seen.add(relative)
    return {
        "entry_count": len(seen),
        "entries_sha256": SOURCE_ENTRIES_SHA256,
        "verification_basis": "pinned_manifest_artifacts_and_canonical_entries_digest",
        "mutable_live_checkout_match_required": False,
    }


def _jsonl_events(
    records: Iterable[dict[str, Any]], *, event_name: str, label: str
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        path = verify_artifact(record, label=f"{label}[{index}]")
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"non-object event in {path}:{line_number}")
                rows.append(row)
        if len(rows) != int(record.get("event_count", -1)):
            raise ValueError(f"{label}[{index}] event count mismatch")
        for row in rows:
            if row.get("event") == event_name:
                selected.append(
                    {
                        **row,
                        "_pipeline_replica": int(record["pipeline_replica"]),
                        "_pipeline_stage": int(record["pipeline_stage"]),
                    }
                )
    if not selected:
        raise ValueError(f"{label} contains no {event_name} events")
    return selected


def _direct_dram_protection_rate(gate: dict[str, Any]) -> tuple[float, int]:
    events = _jsonl_events(
        gate["details"]["stage_aware_peer_hbm"]["files"],
        event_name="pp_copy_completion",
        label="direct_dram_protection",
    )
    grouped: dict[tuple[str, int], dict[int, dict[str, Any]]] = {}
    for row in events:
        if not (
            str(row.get("chunk_id", "")).startswith("dram-")
            and row.get("committed") is True
            and row.get("coordinator_status") == "committed"
            and row.get("transfer_succeeded") is True
            and int(row.get("local_bytes", 0))
            == int(row.get("transferred_bytes", -1))
            and int(row.get("local_bytes", 0)) > 0
            and float(row.get("duration_ms", 0.0)) > 0.0
            and all(int(value) == 0 for value in row.get("put_results", ()))
        ):
            raise ValueError("direct DRAM protection event is not physical/committed")
        key = (str(row["request_id"]), int(row["ordinal"]))
        stage = int(row["_pipeline_stage"])
        grouped.setdefault(key, {})[stage] = row
    rates: list[float] = []
    for key, stages in grouped.items():
        if set(stages) != {0, 1}:
            raise ValueError(f"direct DRAM PP stages are incomplete: {key}")
        replicas = {int(row["_pipeline_replica"]) for row in stages.values()}
        if len(replicas) != 1:
            raise ValueError("direct DRAM PP group crosses pipeline replicas")
        byte_count = sum(int(row["local_bytes"]) for row in stages.values())
        duration_ms = max(float(row["duration_ms"]) for row in stages.values())
        rates.append(byte_count / duration_ms)
    if len(rates) != 4:
        raise ValueError("direct DRAM protection must contain four PP samples")
    return min(rates), len(rates)


def _hbm_recovery_rate(gate: dict[str, Any]) -> tuple[float, int]:
    events = _jsonl_events(
        gate["details"]["stage_aware_recovery"]["files"],
        event_name="hbm_physical_load",
        label="peer_hbm_recovery",
    )
    grouped: dict[tuple[str, int], dict[int, dict[str, Any]]] = {}
    for row in events:
        if not (
            row.get("loaded") is True
            and int(row.get("bytes", 0)) == int(row.get("loaded_bytes", -1))
            and int(row.get("bytes", 0)) > 0
            and float(row.get("duration_ms", 0.0)) > 0.0
            and row.get("load_results")
            and all(row["load_results"])
            and not row.get("missing_storage_keys")
        ):
            raise ValueError("Peer-HBM recovery event is not a physical load")
        key = (str(row["request_id"]), int(row["_pipeline_replica"]))
        grouped.setdefault(key, {})[int(row["_pipeline_stage"])] = row
    rates: list[float] = []
    for key, stages in grouped.items():
        if set(stages) != {0, 1}:
            raise ValueError(f"Peer-HBM recovery PP stages are incomplete: {key}")
        byte_count = sum(int(row["bytes"]) for row in stages.values())
        duration_ms = max(float(row["duration_ms"]) for row in stages.values())
        rates.append(byte_count / duration_ms)
    if len(rates) != 1:
        raise ValueError("Peer-HBM recovery must contain one complete PP sample")
    return min(rates), len(rates)


def _recovery_transport_observation(gate: dict[str, Any]) -> dict[str, Any]:
    details = gate["details"]
    agents = details.get("survivor_hbm_agents")
    if not isinstance(agents, list) or len(agents) != 2:
        raise ValueError("Peer-HBM recovery lacks two survivor agent states")
    agent_stages: set[int] = set()
    for record in agents:
        path = verify_artifact(record, label="survivor Peer-HBM agent state")
        value = load_object(path)
        if value != record.get("value"):
            raise ValueError("survivor Peer-HBM agent embedded state changed")
        data = value.get("data")
        stage = int(record.get("pipeline_stage", -1))
        if not (
            value.get("success") is True
            and isinstance(data, dict)
            and data.get("transfer_protocol") == "tcp"
            and data.get("source_owned_transfers_only") is True
            and data.get("legacy_target_path_enabled") is False
            and stage in {0, 1}
            and str(data.get("pool_id", "")).endswith(f"stage{stage}")
        ):
            raise ValueError(
                "survivor Peer-HBM recovery agent is not TCP/source-owned"
            )
        agent_stages.add(stage)
    if agent_stages != {0, 1}:
        raise ValueError("survivor Peer-HBM recovery agent stages are incomplete")

    starts = _jsonl_events(
        details["stage_aware_recovery"]["files"],
        event_name="hbm_agent_started",
        label="peer_hbm_recovery_agent_start",
    )
    identities = {
        (int(row["_pipeline_replica"]), int(row["_pipeline_stage"]))
        for row in starts
        if row.get("transfer_protocol") == "tcp"
        and row.get("source_owned_transfers_only") is True
    }
    if identities != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise ValueError("Peer-HBM recovery logs do not bind all four agents to TCP")
    return {
        "survivor_agent_state_count": len(agents),
        "event_log_agent_start_count": len(starts),
        "transfer_protocol": "tcp",
        "source_owned_transfers_only": True,
    }


def _peer_transport_observation(gate: dict[str, Any]) -> dict[str, Any]:
    calibration = gate["details"]["physical_calibration"]
    samples: list[dict[str, Any]] = []
    protocols: set[str] = set()
    for node in calibration.get("nodes", []):
        for sample in node.get("samples", []):
            result_record = sample.get("artifacts", {}).get("result_artifact")
            if not isinstance(result_record, dict):
                raise ValueError("Peer-HBM sample lacks its result artifact")
            result_path = verify_artifact(
                result_record, label="Peer-HBM physical sample result"
            )
            result = load_object(result_path)
            source = result.get("source")
            target = result.get("target")
            if not (
                result.get("status") == "passed"
                and result.get("evidence_class")
                == "peer_hbm_two_process_two_gpu_runtime_probe"
                and result.get("transfer_protocol") == "tcp"
                and isinstance(source, dict)
                and isinstance(target, dict)
                and source.get("transfer_protocol") == "tcp"
                and target.get("transfer_protocol") == "tcp"
                and target.get("backing") == "dynamic_registered_tensor"
                and target.get("verified") is True
                and source.get("gpu", {}).get("name") == "NVIDIA GeForce RTX 4090"
                and target.get("gpu", {}).get("name") == "NVIDIA GeForce RTX 4090"
                and source.get("gpu", {}).get("capability") == [8, 9]
                and target.get("gpu", {}).get("capability") == [8, 9]
                and source.get("gpu", {}).get("visible_devices")
                != target.get("gpu", {}).get("visible_devices")
                and int(result.get("payload_bytes", 0)) > 0
            ):
                raise ValueError("Peer-HBM transport sample contract did not pass")
            protocols.add(str(result["transfer_protocol"]))
            samples.append(artifact_record(result_path))
    if len(samples) != 48 or protocols != {"tcp"}:
        raise ValueError("Peer-HBM calibration is not the pinned 48-sample TCP run")
    return {
        "sample_count": len(samples),
        "transfer_protocol": "tcp",
        "physical_path": "registered_gpu_memory_over_mooncake_tcp",
        "direct_cuda_p2p_ready": False,
        "exact_d2d_p2p_contract_ready": False,
        "simulator_rate_default_eligible": False,
        "bandwidth_semantics": (
            "two independent PP-stage registered-HBM-over-TCP lanes"
        ),
        "source_memory": "cuda_device_memory",
        "target_memory": "dynamic_registered_cuda_tensor",
        "cuda_p2p_d2d_verified": False,
        "registered_gpu_memory_over_tcp_verified": True,
        "producer_bandwidth_label_mentions_d2d": "D2D"
        in str(calibration.get("logical_bandwidth_semantics", "")),
        "transport_label_conflict_fail_closed_for_formal_replay": True,
    }


def _verify_failure_acknowledgements(
    report_path: Path, *, request_id: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for cell in ("aegiskv-failed", "no-protection-failed"):
        seen_stages: set[int] = set()
        for node, expected_stage in (("gpu11", 0), ("gpu13", 1)):
            path = report_path.parent / cell / f"failure-ack-{node}.json"
            ack = load_object(path)
            if not (
                ack.get("schema_version") == 1
                and ack.get("event") == "pipeline_stage_process_group_sigkill"
                and ack.get("hostname") == node
                and int(ack.get("pipeline_replica", -1)) == 0
                and int(ack.get("pipeline_stage", -1)) == expected_stage
                and int(ack.get("signal", 0)) == 9
                and ack.get("request_id") == request_id
            ):
                raise ValueError(f"{cell} lacks a valid dual-stage SIGKILL ack")
            seen_stages.add(expected_stage)
            records.append({"cell": cell, **artifact_record(path)})
        if seen_stages != {0, 1}:
            raise ValueError(f"{cell} did not SIGKILL both PP stages")
    return records


def _validate_pair_report(
    path: Path,
    *,
    seed: int,
    strategy: str,
    workload_profile: str,
    failure_delay_ms: float,
    max_tokens: int,
    prompt_word_count: int,
    expected_cached_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = load_object(path)
    order = report.get("details", {}).get("condition_order")
    comparisons = report.get("details", {}).get("comparisons")
    if not (
        report.get("schema_version") == 1
        and report.get("status") == "PASS"
        and report.get("failures") == []
        and report.get("evidence_class")
        == "aegiskv_4090_pp2_paired_targeted_pilot"
        and report.get("design_semantics") == DESIGN_SEMANTICS
        and report.get("result_label") == HARDWARE_LABEL
        and int(report.get("seed", -1)) == seed
        and report.get("formal_ready") is False
        and report.get("paper_ready") is False
        and report.get("preliminary_targeted_pilot") is True
        and isinstance(order, dict)
        and order.get("predeclared_before_execution") is True
        and order.get("strategy") == strategy
        and order.get("workload_profile") == workload_profile
        and float(order.get("failure_delay_ms", -1)) == failure_delay_ms
        and int(order.get("max_tokens", -1)) == max_tokens
        and int(order.get("prompt_word_count", -1)) == prompt_word_count
        and isinstance(comparisons, dict)
        and int(comparisons.get("aegiskv_recovered_cached_tokens", -1))
        == expected_cached_tokens
        and int(comparisons.get("baseline_recovered_cached_tokens", -1)) == 0
    ):
        raise ValueError(f"paired report contract did not pass: {path}")
    details = report["details"]
    aegis_failed = details.get("cell:aegiskv-failed")
    baseline_failed = details.get("cell:no-protection-failed")
    if not isinstance(aegis_failed, dict) or not isinstance(baseline_failed, dict):
        raise ValueError("paired report lacks failed cells")
    request_id = str(aegis_failed.get("request_id") or "")
    if not request_id or baseline_failed.get("request_id") != request_id:
        raise ValueError("paired report request identity mismatch")
    acknowledgements = _verify_failure_acknowledgements(path, request_id=request_id)
    result_record = aegis_failed.get("result")
    if not isinstance(result_record, dict):
        raise ValueError("Aegis failed cell lacks its result artifact")
    result = load_object(verify_artifact(result_record, label="paired Aegis result"))
    if workload_profile == "long-context-recovery-critical-v1":
        stage_recovery = aegis_failed.get("failure_proof", {}).get("stage_recovery")
        survivor_loads = (
            stage_recovery.get("survivor_hbm_loads")
            if isinstance(stage_recovery, dict)
            else None
        )
        if not (
            result.get("response", {}).get("usage", {}).get("prompt_tokens") == 2933
            and result.get("response", {}).get("usage", {}).get(
                "completion_tokens"
            )
            == 192
            and result.get("response", {}).get("usage", {}).get(
                "prompt_tokens_details", {}
            ).get("cached_tokens")
            == 2048
            and aegis_failed.get("recovery", {}).get("peer_hbm_chunk_count") == 4
            and aegis_failed.get("recovery", {}).get("dram_chunk_count") == 0
            and isinstance(survivor_loads, list)
            and {int(row["pipeline_stage"]) for row in survivor_loads} == {0, 1}
            and all(int(row.get("event_count", 0)) > 0 for row in survivor_loads)
            and int(stage_recovery.get("loaded_bytes", 0)) > 0
        ):
            raise ValueError("long-context pair lacks physical two-stage HBM recovery")
        for index, record in enumerate(stage_recovery.get("files", [])):
            verify_artifact(record, label=f"long-context HBM event log {index}")
    manifest_record = details.get("source_config_manifest")
    if not isinstance(manifest_record, dict):
        raise ValueError("paired report lacks source manifest")
    manifest, _entries = _verify_source_manifest(
        manifest_record, label="paired report", expected_entries=None
    )
    return report, {
        "report": artifact_record(path),
        "pair_id": report["pair_id"],
        "seed": seed,
        "strategy": strategy,
        "metrics": dict(comparisons),
        "source_manifest": artifact_record(Path(manifest_record["path"])),
        "source_entries_sha256": manifest["entries_sha256"],
        "failure_acknowledgements": acknowledgements,
    }


def _validate_series(
    series_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    series = load_object(series_path)
    if not (
        series.get("schema_version") == 1
        and series.get("status") == "PASS"
        and series.get("evidence_class")
        == "aegiskv_4090_pp2_paired_targeted_series"
        and series.get("design_semantics") == DESIGN_SEMANTICS
        and series.get("result_label") == HARDWARE_LABEL
        and series.get("source_entries_sha256") == SOURCE_ENTRIES_SHA256
        and series.get("workload_profile")
        == "long-context-recovery-critical-v1"
        and series.get("valid_pair_count") == 4
        and series.get("positive_pair_count") == 4
        and series.get("nonpositive_pair_count") == 0
        and series.get("unique_seeds") == [20260822, 20260823, 20260824, 20260825]
        and series.get("counterbalanced_order")
        == {"aegis-first": 2, "baseline-first": 2}
        and series.get("all_valid_pairs_retained") is True
        and series.get("preliminary_targeted_series") is True
        and series.get("formal_ready") is False
        and series.get("paper_ready") is False
        and series.get("invalid_attempts") == []
    ):
        raise ValueError("long-context paired-series contract did not pass")
    metric = series.get("metrics", {}).get("failure_added_time_saved_ms", {})
    percent = series.get("metrics", {}).get(
        "failure_added_time_saved_percent", {}
    )
    overhead = series.get("metrics", {}).get("healthy_path_overhead_ms", {})
    expected_values = (
        (metric.get("mean"), 139.81402587890625),
        (metric.get("t_95_ci_lower"), 52.445742190714014),
        (metric.get("t_95_ci_upper"), 227.18230956709849),
        (metric.get("exact_bootstrap_95_ci_lower"), 93.1888427734375),
        (metric.get("exact_bootstrap_95_ci_upper"), 186.439208984375),
        (percent.get("mean"), 6.764971463906062),
        (percent.get("t_95_ci_lower"), 2.8520392411405298),
        (percent.get("t_95_ci_upper"), 10.677903686671595),
        (percent.get("exact_bootstrap_95_ci_lower"), 4.652184079591253),
        (percent.get("exact_bootstrap_95_ci_upper"), 8.877758848220871),
        (overhead.get("mean"), 23.875732421875),
    )
    if not all(
        math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-12)
        for observed, expected in expected_values
    ):
        raise ValueError("long-context paired-series metrics changed")
    strategies = {
        20260822: "baseline-first",
        20260823: "aegis-first",
        20260824: "baseline-first",
        20260825: "aegis-first",
    }
    imported_pairs: list[dict[str, Any]] = []
    audit_pairs = series.get("pairs")
    if not isinstance(audit_pairs, list) or len(audit_pairs) != 4:
        raise ValueError("long-context paired-series pair set is incomplete")
    for audit_pair in audit_pairs:
        seed = int(audit_pair.get("seed", -1))
        if seed not in strategies or audit_pair.get("strategy") != strategies[seed]:
            raise ValueError("long-context pair counterbalancing changed")
        report_path = Path(str(audit_pair.get("report_path") or "")).resolve()
        if sha256_file(report_path) != str(audit_pair.get("report_sha256") or ""):
            raise ValueError("long-context pair report hash mismatch")
        report, imported = _validate_pair_report(
            report_path,
            seed=seed,
            strategy=strategies[seed],
            workload_profile="long-context-recovery-critical-v1",
            failure_delay_ms=2000.0,
            max_tokens=192,
            prompt_word_count=2200,
            expected_cached_tokens=2048,
        )
        report_metrics = report["details"]["comparisons"]
        audit_metrics = audit_pair.get("metrics")
        if not isinstance(audit_metrics, dict) or any(
            report_metrics.get(name) != value
            for name, value in audit_metrics.items()
        ):
            raise ValueError("long-context report/series metric mismatch")
        imported_pairs.append(imported)
    return series, imported_pairs


def _validate_full_request_controls(
    baseline_first_path: Path, aegis_first_path: Path
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path, seed, strategy, expected_saved in (
        (baseline_first_path, 20260820, "baseline-first", -43.68701171875),
        (aegis_first_path, 20260821, "aegis-first", 34.28125),
    ):
        report, imported = _validate_pair_report(
            path,
            seed=seed,
            strategy=strategy,
            workload_profile="full-request-v1",
            failure_delay_ms=8000.0,
            max_tokens=512,
            prompt_word_count=720,
            expected_cached_tokens=976,
        )
        saved = float(
            report["details"]["comparisons"]["failure_added_time_saved_ms"]
        )
        if not math.isclose(saved, expected_saved, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("full-request control result changed")
        rows.append(imported)
    mean = sum(float(row["metrics"]["failure_added_time_saved_ms"]) for row in rows) / 2
    if not math.isclose(mean, -4.702880859375, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("full-request control mean changed")
    return {
        "workload_profile": "full-request-v1",
        "interpretation": "decode-heavy near-neutral/unstable control",
        "aggregation_with_long_context_prohibited": True,
        "pair_count": 2,
        "pairs": rows,
        "mean_failure_added_time_saved_ms": mean,
        "advantage_stable": False,
    }


def _validate_figure(
    figure: dict[str, Any], *, paths: InputPaths
) -> dict[str, Any]:
    if not (
        paths.figure_audit.parent.name == "figure-v3"
        and figure.get("schema_version") == 1
        and figure.get("status") == "PASS"
        and figure.get("evidence_class")
        == "aegiskv_4090_pp2_preliminary_series_figure"
        and figure.get("result_label") == HARDWARE_LABEL
        and figure.get("source_entries_sha256") == SOURCE_ENTRIES_SHA256
        and figure.get("workload_profile")
        == "long-context-recovery-critical-v1"
        and figure.get("pair_count") == 4
        and figure.get("all_valid_pairs_retained") is True
        and figure.get("invalid_attempt_count") == 0
        and figure.get("invalid_attempt_note") == INVALID_ATTEMPT_NOTE
        and figure.get("preliminary_targeted_series") is True
        and figure.get("formal_ready") is False
        and figure.get("paper_ready") is False
        and figure.get("positive_is_aegiskv_better") is True
        and figure.get("renderer", {}).get("sha256")
        == "b7326ac961aa02014d1b7eceef724d26e234f99564a00e9ec6f25347a8ca2daf"
        and figure.get("input_audit", {}).get("sha256")
        == EXPECTED_ARTIFACTS["series_audit"][1]
    ):
        raise ValueError("figure-v3 audit contract did not pass")
    expected = {
        "png": _expected_artifact(paths.figure_png, name="figure_png"),
        "pdf": _expected_artifact(paths.figure_pdf, name="figure_pdf"),
        "csv": _expected_artifact(paths.figure_csv, name="figure_csv"),
    }
    for kind, record in expected.items():
        observed = figure.get("artifacts", {}).get(kind)
        if not isinstance(observed, dict):
            raise ValueError(f"figure-v3 lacks {kind}")
        if (
            Path(str(observed.get("path", ""))).resolve()
            != Path(record["path"]).resolve()
            or observed.get("sha256") != record["sha256"]
        ):
            raise ValueError(f"figure-v3 {kind} binding mismatch")
    verify_artifact(figure["renderer"], label="figure-v3 renderer")
    return {
        "audit": _expected_artifact(paths.figure_audit, name="figure_audit"),
        "artifacts": expected,
        "renderer": dict(figure["renderer"]),
        "invalid_attempt_count": 0,
        "invalid_attempt_note": INVALID_ATTEMPT_NOTE,
    }


def validate_snapshot(paths: InputPaths) -> dict[str, Any]:
    """Validate and summarize the immutable RTX4090 PP2 producer snapshot."""

    path_map = paths.as_dict()
    input_records = {
        name: _expected_artifact(path, name=name)
        for name, path in path_map.items()
    }
    gate_values = {
        name: load_object(path_map[name]) for name in GATE_CONTRACTS
    }
    expected_entries: list[dict[str, Any]] | None = None
    canonical_manifest: dict[str, Any] | None = None
    source_manifest_records: dict[str, dict[str, Any]] = {}
    for name, gate in gate_values.items():
        _common_gate(gate, name=name)
        verify_embedded_artifact_tree(gate.get("details", {}), label=name)
        manifest_record = gate.get("details", {}).get("source_config_manifest")
        if not isinstance(manifest_record, dict):
            raise ValueError(f"{name} lacks its source manifest")
        manifest, entries = _verify_source_manifest(
            manifest_record,
            label=name,
            expected_entries=expected_entries,
        )
        if expected_entries is None:
            expected_entries = entries
            canonical_manifest = manifest
        source_manifest_records[name] = artifact_record(
            Path(str(manifest_record["path"]))
        )
    if canonical_manifest is None:
        raise ValueError("no canonical source manifest was found")
    frozen_source = _validate_frozen_source_entries(canonical_manifest)

    topology = gate_values["topology_gate"]
    runtime_nodes = topology["details"].get("runtime_nodes")
    lanes = topology["details"].get("pipeline_transport", {}).get("lanes")
    if not (
        isinstance(runtime_nodes, list)
        and {row.get("hostname") for row in runtime_nodes} == set(NODE_ORDER)
        and all(int(row.get("device_count", 0)) == 2 for row in runtime_nodes)
        and isinstance(lanes, list)
        and len(lanes) == 2
        and {int(row["pipeline_replica"]) for row in lanes} == {0, 1}
        and all(row.get("stage0_node") == "gpu11" for row in lanes)
        and all(row.get("stage1_node") == "gpu13" for row in lanes)
    ):
        raise ValueError("RTX4090 PP2 topology mapping changed")

    serve = gate_values["serve_gate"]
    launch = serve["details"].get("launch_metadata")
    if not isinstance(launch, list) or len(launch) != 2:
        raise ValueError("PP serve gate lacks two stage launch records")
    observed_mapping: set[tuple[str, int, int, int]] = set()
    for node in launch:
        config = node.get("serve_config", {})
        if not (
            config.get("model") == MODEL
            and config.get("pipeline_parallel_size") == 2
            and config.get("tensor_parallel_size") == 1
            and config.get("vllm_internal_data_parallel_size") == 1
            and config.get("dtype") == "bfloat16"
            and config.get("max_model_len") == 4096
        ):
            raise ValueError("PP serve identity changed")
        for process in node.get("processes", []):
            observed_mapping.add(
                (
                    str(node["hostname"]),
                    int(process["pipeline_replica"]),
                    int(process["pipeline_stage"]),
                    int(process["local_gpu_index"]),
                )
            )
    expected_mapping = {
        ("gpu11", 0, 0, 0),
        ("gpu11", 1, 0, 1),
        ("gpu13", 0, 1, 0),
        ("gpu13", 1, 1, 1),
    }
    if observed_mapping != expected_mapping:
        raise ValueError("PP serve process/GPU mapping changed")

    peer = gate_values["peer_hbm_gate"]
    if peer.get("layout") != {
        "tp_size": 1,
        "pp_size": 2,
        "system_dp_size": 2,
    }:
        raise ValueError("Peer-HBM calibration layout changed")
    peer_rate = _positive_number(
        peer["details"]["physical_calibration"].get(
            "conservative_logical_bandwidth_bytes_per_ms"
        ),
        label="Peer-HBM protection rate",
    )
    transport = _peer_transport_observation(peer)

    integration = gate_values["integration_gate"]
    integration_details = integration["details"]
    if not (
        integration.get("independent_source_tier_paths_ready") is True
        and integration.get("forbidden_cross_tier_events_absent") is True
        and integration.get("hbm_admission_policy") == "static"
        and integration.get("dram_admission_policy") == "static"
        and integration.get("algorithm2_policy_evidence") is False
        and integration.get("aimd_policy_evidence") is False
        and integration_details.get("stage_aware_peer_hbm", {}).get(
            "forbidden_cross_tier_event_count"
        )
        == 0
        and int(
            integration_details.get("stage_aware_peer_hbm", {}).get(
                "physical_hbm_copy_bytes", 0
            )
        )
        > 0
        and int(
            integration_details.get("stage_aware_peer_hbm", {}).get(
                "physical_direct_dram_put_bytes", 0
            )
        )
        > 0
    ):
        raise ValueError("independent HBM/direct-DRAM mechanism gate changed")
    direct_dram_rate, direct_dram_samples = _direct_dram_protection_rate(
        integration
    )
    deployment = integration_details["deployment"]["value"]
    if not (
        deployment.get("layout") == LAYOUT
        and deployment.get("model") == MODEL
        and deployment.get("model_revision") == MODEL_REVISION
        and deployment.get("separate_from_v100_campaign") is True
        and deployment.get("connector", {}).get("tier_execution")
        == "source_owned_independent_hbm_d2d_and_dram_d2h"
        and deployment.get("peer_hbm", {}).get("cross_tier_transfer") is False
    ):
        raise ValueError("integration deployment identity changed")

    aimd = gate_values["aimd_gate"]
    if not (
        aimd.get("hbm_only_direct_reclaim_ready") is True
        and aimd.get("independent_dram_bypass_ready") is True
        and aimd.get("all_stage_pause_before_controller_ready") is True
        and aimd.get("all_stage_physical_delete_ready") is True
        and aimd.get("new_hbm_admission_resume_ready") is True
        and aimd.get("forbidden_cross_tier_events_absent") is True
        and aimd.get("aimd_policy_evidence") is True
        and aimd.get("algorithm2_policy_evidence") is False
        and aimd.get("performance_evidence") is False
    ):
        raise ValueError("AIMD direct-reclaim mechanism gate changed")

    recovery = gate_values["recovery_gate"]
    if not (
        recovery.get("whole_pipeline_fail_stop_ready") is True
        and recovery.get("peer_hbm_transfer_ready") is True
        and recovery.get("peer_hbm_recovery_ready") is True
        and recovery.get("peer_hbm_all_stage_commit_ready") is True
        and recovery.get("independent_source_tier_paths_ready") is True
        and recovery.get("forbidden_cross_tier_events_absent") is True
        and recovery.get("algorithm2_policy_evidence") is False
        and recovery.get("aimd_policy_evidence") is False
    ):
        raise ValueError("whole-pipeline physical recovery gate changed")
    hbm_recovery_rate, hbm_recovery_samples = _hbm_recovery_rate(recovery)
    recovery_transport = _recovery_transport_observation(recovery)

    dram = gate_values["dram_gate"]
    dram_summary = dram["details"]["mooncake_dram_calibration"]["summary"]
    if not (
        int(dram_summary.get("sample_count", 0)) == 20
        and int(dram_summary.get("direction_count", 0)) == 2
        and dram["details"]["store_configuration"]["mooncake_store"]["value"].get(
            "protocol"
        )
        == "tcp"
    ):
        raise ValueError("direct Store-DRAM calibration contract changed")
    dram_recovery_rate = _positive_number(
        dram_summary.get("aggregate_effective_load_gb_per_s"),
        label="DRAM recovery GB/s",
    ) * 1_000_000.0

    recompute = gate_values["recompute_gate"]
    recompute_detail = recompute["details"]["recompute_calibration"]
    fit = recompute_detail["pooled_fit"]
    if int(recompute_detail.get("sample_count", 0)) != 30:
        raise ValueError("recompute sample count changed")
    recompute_slope = _positive_number(
        fit.get("recompute_ms_per_token"), label="recompute slope"
    )
    recompute_fixed = _nonnegative_number(
        fit.get("fixed_overhead_ms"), label="recompute fixed overhead"
    )

    series, long_pairs = _validate_series(paths.series_audit)
    figure = _validate_figure(load_object(paths.figure_audit), paths=paths)
    controls = _validate_full_request_controls(
        paths.full_request_baseline_first,
        paths.full_request_aegis_first,
    )
    report_text = paths.preliminary_report.read_text(encoding="utf-8")
    if not (
        "decode-heavy `full-request-v1` profile did not show a stable" in report_text
        and "Only\n`figure-v3/` is valid for reporting." in report_text
        and "registered-HBM protection and recovery over TCP" in report_text
        and "not CUDA P2P/D2D or GDR" in report_text
        and "GPU0<->GPU1 P2P as `NS`" in report_text
        and "torch.cuda.can_device_access_peer(0, 1) == false" in report_text
        and "formal_ready=false" in report_text
        and "paper_ready=false" in report_text
    ):
        raise ValueError("preliminary report omits a required evidence boundary")

    controller = deployment["controller"]
    first_long_report = load_object(
        Path(str(series["pairs"][0]["report_path"])).resolve()
    )
    kv_record = first_long_report["details"]["cell:aegiskv-failed"][
        "launch_metadata"
    ][0]
    kv_launch = load_object(verify_artifact(kv_record, label="paired launch"))
    process = kv_launch["processes"][0]
    kv_config = load_object(
        verify_artifact(process["kv_transfer_config"], label="paired KV config")
    )
    extra = kv_config["kv_connector_extra_config"]
    if not (
        extra.get("aegis_prefix_chunk_tokens") == 512
        and extra.get("aegis_decode_chunk_tokens") == 256
        and extra.get("aegis_protection_mode")
        == "source_owned_independent_hbm_and_dram"
    ):
        raise ValueError("paired-series chunk/protection config changed")

    metrics = {
        "recompute_ms_per_token": recompute_slope,
        "recompute_fixed_overhead_ms": recompute_fixed,
        "peer_hbm_protection_effective_bytes_per_ms": peer_rate,
        "direct_dram_protection_effective_bytes_per_ms": direct_dram_rate,
        "peer_hbm_recovery_effective_bytes_per_ms": hbm_recovery_rate,
        "dram_recovery_effective_bytes_per_ms": dram_recovery_rate,
    }
    if set(metrics) != set(PROFILE_METRICS) or any(
        not math.isfinite(value)
        or (value < 0.0 if name == "recompute_fixed_overhead_ms" else value <= 0.0)
        for name, value in metrics.items()
    ):
        raise ValueError("derived RTX4090 metrics are invalid")
    input_identity = canonical_sha256(
        {
            name: {"relative_path": EXPECTED_ARTIFACTS[name][0], "sha256": record["sha256"]}
            for name, record in sorted(input_records.items())
        }
    )
    return {
        "input_records": input_records,
        "input_identity_sha256": input_identity,
        "source_manifest_records": source_manifest_records,
        "frozen_source": frozen_source,
        "metrics": metrics,
        "transport_observation": transport,
        "recovery_transport_observation": recovery_transport,
        "p2p_topology_diagnostic": {
            "slurm_job_id": "135738",
            "state": "COMPLETED",
            "exit_code": "0:0",
            "elapsed_seconds": 4,
            "nodes": list(NODE_ORDER),
            "allocated_gpu_count": 4,
            "gpu_pair_p2p_status_by_node": {"gpu11": "NS", "gpu13": "NS"},
            "torch_cuda_can_device_access_peer": {
                "gpu11": {"0_to_1": False, "1_to_0": False},
                "gpu13": {"0_to_1": False, "1_to_0": False},
            },
            "evidence_source": "sha256-pinned preliminary report plus bounded read-only live diagnostic",
            "standalone_producer_artifact_available": False,
            "formal_calibration_evidence": False,
        },
        "calibration_sample_counts": {
            "peer_hbm_protection": 48,
            "direct_dram_protection": direct_dram_samples,
            "peer_hbm_recovery": hbm_recovery_samples,
            "dram_recovery": 20,
            "recompute": 30,
        },
        "physical_capacities": {
            "hbm_bytes_per_logical_target": int(
                controller["hbm_capacity_bytes_per_logical_pool"]
            ),
            "hbm_bytes_per_stage_pool": int(
                deployment["peer_hbm"]["capacity_bytes_per_stage_pool"]
            ),
            "dram_bytes_total": int(controller["dram_capacity_bytes"]),
            "dram_bytes_per_logical_target": int(controller["dram_capacity_bytes"])
            // 2,
        },
        "runtime_parameters": {
            "kv_bytes_per_token": 147456,
            "prefix_chunk_tokens": int(extra["aegis_prefix_chunk_tokens"]),
            "decode_chunk_tokens": int(extra["aegis_decode_chunk_tokens"]),
            "pp_size": 2,
            "tp_size": 1,
            "initial_hbm_alpha": float(controller["aimd"]["initial_alpha"]),
            "minimum_hbm_alpha": float(controller["aimd"]["min_alpha"]),
            "maximum_hbm_alpha": float(controller["aimd"]["max_alpha"]),
            "aimd_decrease_factor": float(controller["aimd"]["decrease_factor"]),
            "aimd_increase_step": float(controller["aimd"]["increase_step"]),
            "aimd_stable_windows": int(
                controller["aimd"]["stable_window_threshold"]
            ),
            "event_log_limit": 256,
        },
        "long_context_series": {
            "audit": input_records["series_audit"],
            "workload_profile": "long-context-recovery-critical-v1",
            "prompt_word_count": 2200,
            "actual_prompt_tokens": 2933,
            "max_tokens": 192,
            "failure_delay_ms": 2000.0,
            "pairs": long_pairs,
            "metrics": dict(series["metrics"]),
            "valid_pair_count": 4,
            "all_valid_pairs_retained": True,
        },
        "full_request_control": controls,
        "figure_v3": figure,
    }


def build_profile(paths: InputPaths) -> dict[str, Any]:
    validated = validate_snapshot(paths)
    return {
        "schema_version": 1,
        "evidence_class": PROFILE_EVIDENCE_CLASS,
        "status": "PASS",
        "design_semantics": DESIGN_SEMANTICS,
        "profile_semantics": PROFILE_SEMANTICS,
        "hardware_label": HARDWARE_LABEL,
        "fallback_id": FALLBACK_ID,
        "layout": dict(LAYOUT),
        "node_order": list(NODE_ORDER),
        "model": {
            "name": MODEL,
            "revision": MODEL_REVISION,
            "serving_dtype": "bfloat16",
            "kv_cache_dtype": "bfloat16",
            "block_size": 16,
            "max_model_len": 4096,
            "kv_bytes_per_token": 147456,
            "kv_abi": "vllm-1.2.2+aegiskv.sm89-qwen3-8b-pp2-stage-major-v1",
        },
        "source_lineage": {
            "entries_sha256": SOURCE_ENTRIES_SHA256,
            "frozen_source_verification": validated["frozen_source"],
            "gate_source_manifests": validated["source_manifest_records"],
            "separate_from_v100_campaign": True,
            "separate_from_a100_campaign": True,
        },
        "physical_gates": {
            name: validated["input_records"][name]
            for name in GATE_CONTRACTS
        },
        "metrics": validated["metrics"],
        "runtime_consumption_contract": dict(RUNTIME_CONSUMPTION_CONTRACT),
        "transport_observation": validated["transport_observation"],
        "recovery_transport_observation": validated[
            "recovery_transport_observation"
        ],
        "p2p_topology_diagnostic": validated["p2p_topology_diagnostic"],
        "calibration_sample_counts": validated["calibration_sample_counts"],
        "physical_capacities": validated["physical_capacities"],
        "runtime_parameters": validated["runtime_parameters"],
        "setup_ms": {
            "peer_hbm_protection": 0.0,
            "direct_dram_protection": 0.0,
            "peer_hbm_recovery": 0.0,
            "dram_recovery": 0.0,
        },
        "preliminary_targeted_physical_series": validated[
            "long_context_series"
        ],
        "decode_heavy_control": validated["full_request_control"],
        "reporting_figure": validated["figure_v3"],
        "preliminary_report": validated["input_records"]["preliminary_report"],
        "lineage_inputs": validated["input_records"],
        "lineage_input_sha256": validated["input_identity_sha256"],
        "runtime_authorization": {
            "validation_authorized": True,
            "diagnostic_metadata_runtime_requires_explicit_ack": TCP_DIAGNOSTIC_ACK,
            "formal_replay_authorized": False,
            "paper_performance_claim_authorized": False,
            "reason": (
                "The observed Peer-HBM endpoint memory is registered CUDA memory, "
                "but all 48 transfer samples use Mooncake TCP, not CUDA P2P/D2D."
            ),
        },
        "evidence_boundary": (
            "Independent RTX4090 TP1/PP2/system-DP2 preliminary targeted physical "
            "lineage. The four-pair long-context result and the two-pair decode-"
            "heavy near-neutral control remain separate. This profile is neither "
            "formal nor paper-ready, cannot enter the V100/A100 consumers, and "
            "does not establish CUDA P2P/D2D Peer-HBM transport."
        ),
        "formal_ready": False,
        "paper_ready": False,
    }
