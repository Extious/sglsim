"""Load and validate request/failure/deployment experiment configuration.

The online synthesis runner historically reads only ``client.json`` and
``server.json``.  This adapter keeps that runtime API unchanged while making
the source-of-truth experiment configuration explicitly three-layered:

* ``request.json`` owns workload generation and client concurrency;
* ``failure.json`` owns fault trigger, target, and replay behavior;
* ``deployment.json`` owns the engine topology, timing model, cache, and
  AegisKV policy.

The generated ``client.json`` and ``server.json`` are compatibility artifacts,
never experiment authoring surfaces.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "aegiskv.online-experiment.v1"
DESIGN_SEMANTICS = "source_owned_independent_hbm_dram_v2"

SOURCE_FILENAMES = ("request.json", "failure.json", "deployment.json")

_METADATA_KEYS = {"schema_version", "description"}

# These are workload/client controls only.  Failure and protection-policy
# fields are deliberately absent so a misplaced knob fails closed.
REQUEST_KEYS = {
    "dataset_name",
    "seed",
    "request_rate",
    "max_concurrency",
    "gsp_num_groups",
    "gsp_prompts_per_group",
    "gsp_system_prompt_len",
    "gsp_question_len",
    "gsp_output_len",
    "gsp_range_ratio",
    "gsp_fast_prepare",
    "gsp_send_routing_key",
    "gsp_num_turns",
    "gsp_ordered",
    "gsp_workflow_shape",
    "gsp_stage_context_lens",
    "gsp_stage_long_decode_agent_index",
    "gsp_long_decode_output_multiplier",
    "num_prompts",
    "gsp_critical_agent_ids",
    "gsp_critical_agent_ratio",
    "gsp_max_workflow_concurrency",
    "trace_jsonl_path",
    "trace_limit",
}

FAILURE_KEYS = {
    "enabled",
    "failover_replay",
    "failover_replay_critical_only",
    "injection",
    "retry_rank_offset",
    "retry_policy",
    "rate_per_window",
    "after_output_tokens",
    "injection_count",
    "target_stage_id",
    "target_stage_local_agent_index",
    "target_turn_idx",
    "dp_sweep_enabled",
    "dp_sweep_hold_s",
    "dp_sweep_gap_s",
    "dp_sweep_initial_delay_s",
    "dp_sweep_injections_per_batch",
    "dp_sweep_batch_size",
    "dp_sweep_batch_gap_s",
    "dp_sweep_round1_enabled",
    "dp_sweep_round2_enabled",
    "dp_sweep_round2_injections_per_batch",
    "dp_sweep_round2_batch_gap_s",
    "dp_sweep_precovered_agent_slots",
    "dp_sweep_precovered_workflows",
    "agent_slot_sweep_enabled",
    "agent_slot_sweep_hold_s",
    "agent_slot_sweep_gap_s",
    "agent_slot_sweep_initial_delay_s",
    "agent_slot_sweep_min_injections",
}

AEGISKV_KEYS = {
    "design_semantics",
    "policy",
    "ablation",
    "gpu_budget_fraction",
    "critical_pressure",
    "alpha_policy_path",
    "online_alpha_adapter",
    "backup_kv",
}

_FAILURE_TO_CLIENT = {
    "failover_replay": "gsp_failover_replay",
    "failover_replay_critical_only": "gsp_failover_replay_critical_only",
    "injection": "gsp_failover_injection",
    "retry_rank_offset": "gsp_failover_retry_rank_offset",
    "retry_policy": "gsp_failover_retry_policy",
    "rate_per_window": "gsp_failure_rate_per_window",
    "after_output_tokens": "gsp_failure_after_output_tokens",
    "injection_count": "gsp_failure_injection_count",
    "target_stage_id": "gsp_failure_target_stage_id",
    "target_stage_local_agent_index": "gsp_failure_target_stage_local_agent_index",
    "target_turn_idx": "gsp_failure_target_turn_idx",
    "dp_sweep_enabled": "gsp_dp_failure_sweep",
    "dp_sweep_hold_s": "gsp_dp_failure_sweep_hold_s",
    "dp_sweep_gap_s": "gsp_dp_failure_sweep_gap_s",
    "dp_sweep_initial_delay_s": "gsp_dp_failure_sweep_initial_delay_s",
    "dp_sweep_injections_per_batch": "gsp_dp_failure_sweep_injections_per_batch",
    "dp_sweep_batch_size": "gsp_dp_failure_sweep_batch_size",
    "dp_sweep_batch_gap_s": "gsp_dp_failure_sweep_batch_gap_s",
    "dp_sweep_round1_enabled": "gsp_dp_failure_sweep_round1_enabled",
    "dp_sweep_round2_enabled": "gsp_dp_failure_sweep_round2_enabled",
    "dp_sweep_round2_injections_per_batch": (
        "gsp_dp_failure_sweep_round2_injections_per_batch"
    ),
    "dp_sweep_round2_batch_gap_s": "gsp_dp_failure_sweep_round2_batch_gap_s",
    "dp_sweep_precovered_agent_slots": (
        "gsp_dp_failure_sweep_precovered_agent_slots"
    ),
    "dp_sweep_precovered_workflows": (
        "gsp_dp_failure_sweep_precovered_workflows"
    ),
    "agent_slot_sweep_enabled": "gsp_agent_slot_failure_sweep",
    "agent_slot_sweep_hold_s": "gsp_agent_slot_sweep_hold_s",
    "agent_slot_sweep_gap_s": "gsp_agent_slot_sweep_gap_s",
    "agent_slot_sweep_initial_delay_s": "gsp_agent_slot_sweep_initial_delay_s",
    "agent_slot_sweep_min_injections": "gsp_agent_slot_sweep_min_injections",
}

_AEGISKV_TO_CLIENT = {
    "policy": "gsp_aegis_policy",
    "ablation": "gsp_aegis_ablation",
    "gpu_budget_fraction": "gsp_aegis_gpu_budget_fraction",
    "critical_pressure": "gsp_aegis_critical_pressure",
    "alpha_policy_path": "gsp_alpha_policy_path",
    "online_alpha_adapter": "gsp_aegis_online_alpha_adapter",
    "backup_kv": "gsp_backup_kv",
}


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing experiment configuration: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a JSON object: {path}")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path.name} must declare schema_version={SCHEMA_VERSION!r}"
        )
    return value


def _reject_unknown(
    raw: Mapping[str, Any], allowed: set[str], *, layer: str
) -> None:
    unknown = sorted(set(raw) - allowed - _METADATA_KEYS)
    if unknown:
        raise ValueError(f"unknown {layer} configuration keys: {unknown}")


def _without_metadata(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in raw.items() if key not in _METADATA_KEYS}


def _resolve_optional_path(value: Any, config_dir: Path) -> Any:
    if not isinstance(value, str) or not value.strip():
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((config_dir / path).resolve())


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class ThreeLayerExperimentConfig:
    config_dir: Path
    request: dict[str, Any]
    failure: dict[str, Any]
    deployment: dict[str, Any]

    @property
    def name(self) -> str:
        raw = str(self.deployment.get("experiment_name", self.config_dir.name))
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-.")
        return slug or self.config_dir.name

    @property
    def aegiskv(self) -> dict[str, Any]:
        value = self.deployment.get("aegiskv", {})
        return dict(value) if isinstance(value, dict) else {}

    def effective_client(self) -> dict[str, Any]:
        client = _without_metadata(self.request)
        if client.get("trace_jsonl_path"):
            client["trace_jsonl_path"] = _resolve_optional_path(
                client["trace_jsonl_path"], self.config_dir
            )

        enabled = bool(self.failure.get("enabled", False))
        for source_key, target_key in _FAILURE_TO_CLIENT.items():
            value = self.failure.get(source_key)
            if value is not None:
                client[target_key] = value

        if not enabled:
            # These switches are the only failure controls with active defaults.
            # Force them off even if a copied template contains stale values.
            client["gsp_failover_replay"] = False
            client["gsp_dp_failure_sweep"] = False
            client["gsp_agent_slot_failure_sweep"] = False
            for key in (
                "gsp_failure_rate_per_window",
                "gsp_failure_after_output_tokens",
                "gsp_failure_injection_count",
                "gsp_failure_target_stage_id",
                "gsp_failure_target_stage_local_agent_index",
                "gsp_failure_target_turn_idx",
            ):
                client.pop(key, None)

        aegiskv = self.aegiskv
        for source_key, target_key in _AEGISKV_TO_CLIENT.items():
            value = aegiskv.get(source_key)
            if value is None:
                continue
            if source_key == "alpha_policy_path":
                value = _resolve_optional_path(value, self.config_dir)
            client[target_key] = value
        return client

    def effective_server(self) -> dict[str, Any]:
        return _without_metadata(self.deployment)


def _validate_request(raw: Mapping[str, Any]) -> None:
    _reject_unknown(raw, REQUEST_KEYS, layer="request")
    if "dataset_name" not in raw:
        raise ValueError("request.json must define dataset_name")
    request_rate = raw.get("request_rate", "inf")
    if not (
        request_rate == float("inf")
        or (isinstance(request_rate, str) and request_rate.lower() in {"inf", "infinity"})
    ):
        raise ValueError(
            "online synthesis request_rate must be 'inf'; use max_concurrency"
        )
    if raw.get("max_concurrency") is not None and int(raw["max_concurrency"]) <= 0:
        raise ValueError("request max_concurrency must be positive or null")


def _failure_has_trigger(raw: Mapping[str, Any]) -> bool:
    return bool(
        raw.get("rate_per_window") is not None
        or raw.get("after_output_tokens") is not None
        or raw.get("injection_count") is not None
        or raw.get("dp_sweep_enabled", False)
        or raw.get("agent_slot_sweep_enabled", False)
    )


def _validate_failure(raw: Mapping[str, Any]) -> None:
    _reject_unknown(raw, FAILURE_KEYS, layer="failure")
    if not isinstance(raw.get("enabled"), bool):
        raise ValueError("failure.json must define boolean enabled")
    if raw.get("dp_sweep_enabled") and raw.get("agent_slot_sweep_enabled"):
        raise ValueError("dp_sweep and agent_slot_sweep cannot both be enabled")
    if raw.get("enabled") and not _failure_has_trigger(raw):
        raise ValueError("enabled failure configuration has no trigger")
    for key in (
        "after_output_tokens",
        "injection_count",
        "retry_rank_offset",
        "dp_sweep_injections_per_batch",
        "dp_sweep_batch_size",
        "agent_slot_sweep_min_injections",
    ):
        value = raw.get(key)
        if value is not None and int(value) < 0:
            raise ValueError(f"failure {key} must be non-negative or null")


def _validate_deployment(raw: Mapping[str, Any]) -> None:
    for required in ("experiment_name", "model_path", "scheduler", "aegiskv"):
        if required not in raw:
            raise ValueError(f"deployment.json must define {required}")
    scheduler = raw.get("scheduler")
    if not isinstance(scheduler, dict):
        raise ValueError("deployment scheduler must be a JSON object")
    for size_key in ("dp_size", "tp_size", "pp_size"):
        if int(scheduler.get(size_key, 1)) <= 0:
            raise ValueError(f"deployment scheduler.{size_key} must be positive")
    aegiskv = raw.get("aegiskv")
    if not isinstance(aegiskv, dict):
        raise ValueError("deployment aegiskv must be a JSON object")
    unknown = sorted(set(aegiskv) - AEGISKV_KEYS)
    if unknown:
        raise ValueError(f"unknown deployment aegiskv keys: {unknown}")
    if aegiskv.get("design_semantics") != DESIGN_SEMANTICS:
        raise ValueError(
            "deployment aegiskv.design_semantics must select "
            f"{DESIGN_SEMANTICS!r}"
        )
    fraction = aegiskv.get("gpu_budget_fraction")
    if fraction is not None and not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("aegiskv gpu_budget_fraction must be in [0, 1]")


def load_experiment_config(config_dir: Path) -> ThreeLayerExperimentConfig:
    config_dir = Path(config_dir).resolve()
    request = _read_json_object(config_dir / "request.json")
    failure = _read_json_object(config_dir / "failure.json")
    deployment = _read_json_object(config_dir / "deployment.json")
    _validate_request(request)
    _validate_failure(failure)
    _validate_deployment(deployment)
    return ThreeLayerExperimentConfig(
        config_dir=config_dir,
        request=request,
        failure=failure,
        deployment=deployment,
    )


def materialize_runtime_config(
    config: ThreeLayerExperimentConfig, output_dir: Path
) -> Path:
    """Write the two compatibility files consumed by synthesis_requests."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "client.json", config.effective_client())
    _write_json(output_dir / "server.json", config.effective_server())
    return output_dir


def write_campaign_snapshot(
    config: ThreeLayerExperimentConfig,
    campaign_dir: Path,
    *,
    status: str = "prepared",
    runtime_run_dir: Path | None = None,
    error: str = "",
) -> Path:
    """Persist source and effective configuration plus a hash manifest."""

    campaign_dir = Path(campaign_dir)
    source_dir = campaign_dir / "source_config"
    effective_dir = campaign_dir / "effective_config"
    for filename, value in (
        ("request.json", config.request),
        ("failure.json", config.failure),
        ("deployment.json", config.deployment),
    ):
        _write_json(source_dir / filename, value)
    materialize_runtime_config(config, effective_dir)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": config.name,
        "design_semantics": config.aegiskv.get("design_semantics", ""),
        "execution_mode": "online_wall_clock_http",
        "physical_result": False,
        "status": str(status),
        "source_config_dir": str(config.config_dir),
        "source_sha256": {
            "request.json": _sha256(config.request),
            "failure.json": _sha256(config.failure),
            "deployment.json": _sha256(config.deployment),
        },
        "effective_sha256": {
            "client.json": _sha256(config.effective_client()),
            "server.json": _sha256(config.effective_server()),
        },
        "runtime_entrypoint": "benchmark.synthesis_requests.run.run_experiment",
        "runtime_run_dir": str(runtime_run_dir) if runtime_run_dir else "",
        "error": str(error),
    }
    path = campaign_dir / "experiment_manifest.json"
    _write_json(path, manifest)
    return path
