"""Compile v3 deployment/failure layers into the offline physical runtime.

The frozen v3 JSON files are an experiment *plan*, not sufficient authority to
invent model revisions, hardware calibration, failure targets, or repair-delay
semantics.  This module therefore has two deliberately separate entry points:

* readiness functions enumerate every missing or unsupported binding as a
  structured error; and
* compilation functions run only after that report is empty.

Successful deployment compilation produces the exact DP/PP/TP-to-GPU map, one
node-local DRAM pool per node, a complete transfer-resource catalogue, the
configured copy-fabric model, and the constructor arguments consumed by
``OfflineSchedulerCluster``.  Successful failure compilation produces only
``schedule_fail_stop`` calls.  It never falls back to the legacy soft metadata
failure API.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from AegisKV.experiments.src.global_des_config import (
    GB,
    GIB,
    GROUP_IDS,
    GroupConfig,
    SuiteConfig,
    build_offline_physical_topology,
    config_hash,
)
from sglsim.srt.simulation.aegis_independent_tiers import DESIGN_SEMANTICS
from sglsim.srt.simulation.global_des import GlobalDES, TransferResourceId
from sglsim.srt.simulation.offline_failure_domain import (
    FailureTargetNotRepresentableError,
    assert_current_cluster_representable,
    resolve_offline_failure_target,
)
from sglsim.srt.simulation.offline_aegis_transfer_bridge import (
    CopyFabricBandwidths,
    CopyFabricConfig,
    CopyFabricSetupLatencies,
)
from sglsim.srt.simulation.offline_resource_bridge import OfflinePhysicalTopology


COMPILER_SCHEMA = "aegiskv.global-des-runtime-compilation.v1"
SUPPORTED_GROUPS = frozenset(GROUP_IDS[1:])
_REVISION_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ACTIVE_EXPOSURE_PLAN_SCHEMA = "aegiskv.policy-independent-failure-plan.v1"


@dataclass(frozen=True)
class RuntimeReadinessError:
    """One machine-readable reason a cell cannot be launched."""

    code: str
    layer: str
    config_path: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "layer": self.layer,
            "config_path": self.config_path,
            "message": self.message,
        }


@dataclass(frozen=True)
class RuntimeReadiness:
    errors: tuple[RuntimeReadinessError, ...]

    @property
    def ready(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "errors": [error.as_dict() for error in self.errors],
        }


class RuntimeNotReadyError(RuntimeError):
    """Raised before any mutable runtime object is created."""

    def __init__(self, errors: Sequence[RuntimeReadinessError]):
        self.errors = tuple(errors)
        super().__init__(
            "global-DES runtime is not ready: "
            + "; ".join(
                f"{error.code}@{error.layer}{error.config_path}: {error.message}"
                for error in self.errors
            )
        )

    def as_dict(self) -> dict[str, object]:
        return RuntimeReadiness(self.errors).as_dict()


@dataclass(frozen=True)
class RuntimeBindings:
    """Run-owned identities that the frozen plan intentionally does not guess.

    ``hardware_timing_class`` is exactly ``calibrated`` or ``modeled``.  A
    modeled run additionally requires ``allow_modeled_hardware=True`` and is
    permanently labelled modeled in the compiled provenance.
    """

    model_revisions: Mapping[str, str] = field(default_factory=dict)
    simulator_revision: str = ""
    hardware_timing_class: str = ""
    hardware_timing_source: str = ""
    hardware_timing_sha256: str = ""
    allow_modeled_hardware: bool = False
    failure_target_selector: str = ""
    repair_delay_policy: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_revisions",
            MappingProxyType(dict(self.model_revisions)),
        )


@dataclass(frozen=True)
class TransferResourceCatalog:
    """All serial resources addressable by configured physical copy paths."""

    resources: tuple[TransferResourceId, ...]
    path_contracts: Mapping[str, tuple[str, ...]]
    profiles: Mapping[str, Mapping[str, int | str]]

    def as_dict(self) -> dict[str, object]:
        return {
            "resources": [resource.key for resource in self.resources],
            "path_contracts": {
                key: list(value) for key, value in sorted(self.path_contracts.items())
            },
            "profiles": {
                key: dict(value) for key, value in sorted(self.profiles.items())
            },
        }


@dataclass(frozen=True)
class ProtectionPolicyContract:
    policy: str
    tiers: tuple[str, ...]
    request_policy: str
    algorithm_variant: str
    replica_placement: str
    request_simulation_defaults: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "tiers": list(self.tiers),
            "request_policy": self.request_policy,
            "algorithm_variant": self.algorithm_variant,
            "replica_placement": self.replica_placement,
            "request_simulation_defaults": dict(self.request_simulation_defaults),
        }


@dataclass(frozen=True)
class CompiledDeployment:
    group: str
    model_name: str
    model_id: str
    model_revision: str
    node_count: int
    dp_size: int
    pp_size: int
    tp_size: int
    topology: OfflinePhysicalTopology
    transfer_resources: TransferResourceCatalog
    copy_fabric: CopyFabricConfig
    policy: ProtectionPolicyContract
    hbm_aegis_budget_bytes_per_gpu: int
    dram_aegis_budget_bytes_per_node: int
    dram_aegis_budget_tokens_per_node: int
    dram_aegis_unallocatable_tail_bytes: int
    simulator_config_overlay: Mapping[str, object]
    provenance: Mapping[str, object]

    @property
    def gpu_count(self) -> int:
        return len(self.topology.gpus)

    def new_global_des(self, *, start_ns: int = 0) -> GlobalDES:
        """Return an unregistered kernel suitable for OfflineSchedulerCluster.

        GPU/DRAM registration remains owned by ``OfflineResourceBridge`` inside
        the cluster constructor.  Transfer calendars are seeded at zero so the
        complete deployment resource graph is visible before the first copy.
        """

        des = GlobalDES(start_ns=start_ns)
        des.resource_busy_until_ns.update(
            {resource: int(start_ns) for resource in self.transfer_resources.resources}
        )
        return des

    def offline_cluster_kwargs(self, *, start_ns: int = 0) -> dict[str, object]:
        return {
            "global_des": self.new_global_des(start_ns=start_ns),
            "physical_topology": self.topology,
        }

    def bind_started_cluster(self, cluster: object) -> dict[str, object]:
        """Install the configured fabric before any physical copy is planned."""

        des = getattr(cluster, "global_des", None)
        bridge = getattr(cluster, "resource_bridge", None)
        if not isinstance(des, GlobalDES) or bridge is None:
            raise TypeError("cluster must be an OfflineSchedulerCluster with physical topology")
        topology = getattr(bridge, "topology", None)
        if not isinstance(topology, OfflinePhysicalTopology):
            raise TypeError("cluster physical resource bridge has no topology")
        if topology.as_dict() != self.topology.as_dict():
            raise ValueError("cluster topology differs from compiled deployment")
        for resource in self.transfer_resources.resources:
            des.resource_busy_until_ns.setdefault(resource, des.now_ns)
        aegis_bridge = getattr(cluster, "aegis_transfer_bridge", None)
        hicache_bridge = getattr(cluster, "hicache_transfer_bridge", None)
        if hicache_bridge is None:
            raise RuntimeError(
                "formal global-DES runner requires ordinary HiCache adapter"
            )
        if tuple(getattr(hicache_bridge, "transfers", {}).values()):
            raise RuntimeError(
                "copy fabric must be bound before the first ordinary HiCache job"
            )
        hicache_bridge.fabric = self.copy_fabric
        dram_quota_bound = "dram" not in self.policy.tiers
        if "dram" in self.policy.tiers:
            quota = getattr(cluster, "aegis_per_node_dram_quota", None)
            quota_dict = getattr(quota, "as_dict", None)
            if not callable(quota_dict):
                raise RuntimeError(
                    "compiled DRAM policy requires explicit OfflineCluster quota binding"
                )
            bound = quota_dict()
            if (
                int(bound.get("quota_bytes", -1))
                != self.dram_aegis_budget_bytes_per_node
                or int(bound.get("quota_tokens", -1))
                != self.dram_aegis_budget_tokens_per_node
                or not bool(bound.get("explicit", False))
            ):
                raise ValueError(
                    "OfflineCluster per-node Aegis DRAM quota differs from deployment"
                )
            dram_quota_bound = True
        if self.policy.tiers:
            if aegis_bridge is None:
                raise RuntimeError("compiled protected policy requires physical Aegis bridge")
            if tuple(getattr(aegis_bridge, "jobs", ())):
                raise RuntimeError("copy fabric must be bound before the first Aegis job")
            aegis_bridge.fabric = self.copy_fabric
        return {
            "schema_version": COMPILER_SCHEMA,
            "group": self.group,
            "copy_fabric_bound": bool(self.policy.tiers),
            "ordinary_hicache_copy_fabric_bound": True,
            "per_node_dram_quota_bound": dram_quota_bound,
            "resource_count": len(self.transfer_resources.resources),
            "global_des_now_ns": des.now_ns,
        }

    def as_dict(self) -> dict[str, object]:
        fabric_resource = TransferResourceId(
            "fabric", self.copy_fabric.fabric_id
        ).key
        nodes = {node.node_id: node for node in self.topology.nodes}
        rank_resource_bindings = []
        for spec in sorted(self.topology.gpus, key=lambda item: item.gpu):
            gpu = spec.gpu
            owned = [
                resource.key
                for resource in self.transfer_resources.resources
                if resource.owner in {gpu.key, gpu.node_id}
                or resource.key == fabric_resource
            ]
            rank_resource_bindings.append(
                {
                    "gpu": gpu.as_dict(),
                    "node_dram_id": gpu.node_id,
                    "node_dram": nodes[gpu.node_id].as_dict(),
                    "transfer_resources": owned,
                }
            )
        bandwidths = self.copy_fabric.bandwidths
        setup = self.copy_fabric.setup
        return {
            "schema_version": COMPILER_SCHEMA,
            "group": self.group,
            "model": {
                "name": self.model_name,
                "model_id": self.model_id,
                "revision": self.model_revision,
            },
            "parallelism": {
                "node_count": self.node_count,
                "dp_size": self.dp_size,
                "pp_size": self.pp_size,
                "tp_size": self.tp_size,
            },
            "topology": self.topology.as_dict(),
            "rank_resource_bindings": rank_resource_bindings,
            "transfer_resources": self.transfer_resources.as_dict(),
            "copy_fabric": {
                "fabric_id": self.copy_fabric.fabric_id,
                "bandwidth_bytes_per_s": {
                    name: int(getattr(bandwidths, name))
                    for name in bandwidths.__dataclass_fields__
                },
                "setup_ns": {
                    name: int(getattr(setup, name))
                    for name in setup.__dataclass_fields__
                },
            },
            "policy": self.policy.as_dict(),
            "budgets": {
                "hbm_aegis_bytes_per_gpu": self.hbm_aegis_budget_bytes_per_gpu,
                "dram_aegis_bytes_per_node": self.dram_aegis_budget_bytes_per_node,
                "dram_aegis_tokens_per_node": self.dram_aegis_budget_tokens_per_node,
                "dram_unallocatable_tail_bytes_per_node": (
                    self.dram_aegis_unallocatable_tail_bytes
                ),
            },
            "simulator_config_overlay": copy.deepcopy(dict(self.simulator_config_overlay)),
            "provenance": copy.deepcopy(dict(self.provenance)),
        }


@dataclass(frozen=True)
class HardFailureEvent:
    event_id: str
    scenario: str
    inject_at_ns: int
    repair_at_ns: int
    failure_scope: str
    dp_rank: int | None = None
    node_id: str | None = None
    resolved_physical_target: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @property
    def duration_ns(self) -> int:
        return self.repair_at_ns - self.inject_at_ns

    def as_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "scenario": self.scenario,
            "failure_mode": "hard_fail_stop",
            "inject_at_ns": self.inject_at_ns,
            "repair_at_ns": self.repair_at_ns,
            "duration_ns": self.duration_ns,
            "failure_scope": self.failure_scope,
            "dp_rank": self.dp_rank,
            "node_id": self.node_id,
            "resolved_physical_target": copy.deepcopy(
                dict(self.resolved_physical_target)
            ),
        }


@dataclass(frozen=True)
class CompiledFailureTimeline:
    group: str
    scenario: str
    events: tuple[HardFailureEvent, ...]
    metadata: Mapping[str, object]

    def install(self, cluster: object) -> int:
        """Schedule hard process/resource loss; soft metadata failure is illegal."""

        schedule_hard = getattr(cluster, "schedule_fail_stop", None)
        if not callable(schedule_hard):
            raise TypeError(
                "runtime lacks schedule_fail_stop; refusing legacy soft metadata failure"
            )
        for event in self.events:
            kwargs: dict[str, object] = {
                "event_id": event.event_id,
                "offset_s": event.inject_at_ns / 1_000_000_000.0,
                "duration_s": event.duration_ns / 1_000_000_000.0,
                "failure_scope": event.failure_scope,
                "payload": {
                    "compiled_failure_schema": COMPILER_SCHEMA,
                    "scenario": event.scenario,
                    "repair_delay_policy": self.metadata["repair_delay_policy"],
                    "trigger_mode": self.metadata.get("trigger_mode"),
                    "trigger_normalization": copy.deepcopy(
                        self.metadata.get("trigger_normalization")
                    ),
                    "active_exposure_plan": copy.deepcopy(
                        self.metadata.get("active_exposure_plan")
                    ),
                    "shadow_plan_sha256": self.metadata.get(
                        "shadow_plan_sha256"
                    ),
                    "anchor_request_id": self.metadata.get(
                        "anchor_request_id"
                    ),
                    "anchor_source_dp_rank": self.metadata.get(
                        "anchor_source_dp_rank"
                    ),
                    "anchor_source_node_id": self.metadata.get(
                        "anchor_source_node_id"
                    ),
                    "soft_metadata_failure_forbidden": True,
                    "resolved_physical_target": copy.deepcopy(
                        dict(event.resolved_physical_target)
                    ),
                },
            }
            if event.failure_scope == "node":
                kwargs["node_id"] = event.node_id
            else:
                kwargs["dp_rank"] = event.dp_rank
            schedule_hard(**kwargs)
        return len(self.events)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": COMPILER_SCHEMA,
            "group": self.group,
            "scenario": self.scenario,
            "failure_mode": "hard_fail_stop",
            "events": [event.as_dict() for event in self.events],
            "metadata": copy.deepcopy(dict(self.metadata)),
        }


@dataclass(frozen=True)
class _FailureDeploymentShape:
    """Read-only dimensions used to audit failure JSON before evidence binds."""

    node_count: int
    dp_size: int
    pp_size: int
    tp_size: int
    gpu_count: int


def _issue(
    errors: list[RuntimeReadinessError],
    code: str,
    layer: str,
    path: str,
    message: str,
) -> None:
    errors.append(RuntimeReadinessError(code, layer, path, message))


def _axis_errors(
    resolved: Mapping[str, Any],
    axes: Mapping[str, Any],
    *,
    layer: str,
) -> list[RuntimeReadinessError]:
    errors: list[RuntimeReadinessError] = []
    declared = resolved.get("axes")
    if not isinstance(declared, Mapping):
        _issue(errors, "axes_missing", layer, "#/axes", "resolved axes object required")
        return errors
    for name, domain in declared.items():
        path = f"#/axes/{name}"
        if name not in axes:
            _issue(errors, "axis_value_missing", layer, path, "selected axis value required")
        elif not isinstance(domain, Sequence) or isinstance(domain, (str, bytes)):
            _issue(errors, "axis_domain_invalid", layer, path, "axis domain must be a sequence")
        elif axes[name] not in domain:
            _issue(
                errors,
                "axis_value_out_of_domain",
                layer,
                path,
                f"selected value {axes[name]!r} is not declared",
            )
    return errors


def _resolved_group(resolved: Mapping[str, Any], layer: str) -> str:
    if resolved.get("schema_version") != "aegiskv.global-des-layer.v3":
        return ""
    if resolved.get("layer") != layer:
        return ""
    return str(resolved.get("group") or "")


def _selected(
    resolved: Mapping[str, Any], axes: Mapping[str, Any], name: str, default: Any
) -> Any:
    return axes.get(name, default) if name in resolved.get("axes", {}) else default


def _resolved_model_revision(
    fixed: Mapping[str, Any],
    model_name: str,
    bindings: RuntimeBindings,
) -> tuple[str, str, bool]:
    revisions = (
        fixed.get("provenance", {})
        .get("model_revisions", {})
        .get("revisions", {})
    )
    configured = str(revisions.get(model_name) or "")
    runtime = str(bindings.model_revisions.get(model_name) or "")
    conflict = bool(configured and runtime and configured != runtime)
    if runtime:
        return runtime, "runtime_binding", conflict
    if configured:
        return configured, "frozen_deployment_json", False
    return "", "missing", False


def _resolved_hardware_timing(
    fixed: Mapping[str, Any], bindings: RuntimeBindings
) -> dict[str, object]:
    if bindings.hardware_timing_class:
        timing_class = str(bindings.hardware_timing_class)
        return {
            "timing_class": timing_class,
            "evidence_class": (
                "calibrated"
                if timing_class == "calibrated"
                else "modeled_not_calibrated"
            ),
            "source": str(bindings.hardware_timing_source),
            "sha256": str(bindings.hardware_timing_sha256),
            "modeled_authorized": bool(bindings.allow_modeled_hardware),
            "binding_source": "runtime_binding",
            "assumption_id": "",
        }
    hardware = fixed.get("provenance", {}).get("hardware_timing", {})
    evidence = str(hardware.get("evidence_class") or "")
    timing_class = "modeled" if evidence == "modeled_not_calibrated" else ""
    assumption_id = str(hardware.get("assumption_id") or "")
    source = str(hardware.get("source") or "")
    integrity_hash = config_hash(
        {
            "cluster": fixed.get("cluster", {}),
            "copy_resources": fixed.get("copy_resources", {}),
        }
    )
    return {
        "timing_class": timing_class,
        "evidence_class": evidence,
        "source": source,
        "sha256": integrity_hash,
        "modeled_authorized": bool(
            hardware.get("modeled_execution_authorized", False)
        ),
        "binding_source": "frozen_deployment_json",
        "assumption_id": assumption_id,
    }


def _policy_contract(
    fixed: Mapping[str, Any], axes: Mapping[str, Any], resolved: Mapping[str, Any]
) -> ProtectionPolicyContract:
    policy = str(_selected(resolved, axes, "policy", fixed["aegiskv"]["central_policy"]))
    mapping = {
        "recompute": ((), "aegiskv_no_backup"),
        "no_protection_control": ((), "aegiskv_no_backup"),
        "host_checkpoint": (("dram",), "aegiskv_l2_only"),
        "neighbor_replica": (("hbm",), "aegiskv_l1_only"),
        "aegiskv": (("hbm", "dram"), "aegiskv_l1_l2"),
    }
    tiers, request_policy = mapping[policy]
    algorithm = str(_selected(resolved, axes, "algorithm_variant", "full"))
    placement = str(_selected(resolved, axes, "replica_placement", "topology_aware_runtime"))
    defaults: dict[str, object] = {
        "aegis_policy": request_policy,
        "backup_kv": bool(tiers),
    }
    if placement != "topology_aware_runtime":
        defaults["aegis_replica_placement"] = placement
    if algorithm == "static_placement":
        defaults["aegis_static_placement"] = True
    if algorithm == "no_algorithm_1":
        defaults["aegis_disable_slack_ordering"] = True
    if algorithm == "no_algorithm_3":
        defaults["aegis_disable_aimd"] = True
    return ProtectionPolicyContract(
        policy=policy,
        tiers=tuple(tiers),
        request_policy=request_policy,
        algorithm_variant=algorithm,
        replica_placement=placement,
        request_simulation_defaults=MappingProxyType(defaults),
    )


def deployment_layer_readiness(
    resolved_deployment: Mapping[str, Any],
    axes: Mapping[str, Any],
    bindings: RuntimeBindings | None = None,
) -> RuntimeReadiness:
    bindings = bindings or RuntimeBindings()
    errors = _axis_errors(resolved_deployment, axes, layer="deployment")
    group = _resolved_group(resolved_deployment, "deployment")
    if not group:
        _issue(errors, "layer_mismatch", "deployment", "#/layer", "resolved deployment v3 layer required")
        return RuntimeReadiness(tuple(errors))
    if group not in SUPPORTED_GROUPS:
        _issue(errors, "unsupported_group", "deployment", "#/group", "compiler scope is P1-P12")
    fixed = resolved_deployment.get("fixed")
    if not isinstance(fixed, Mapping):
        _issue(errors, "fixed_missing", "deployment", "#/fixed", "resolved fixed object required")
        return RuntimeReadiness(tuple(errors))

    model_name = str(_selected(resolved_deployment, axes, "model", fixed["central_model"]))
    model = fixed["models"].get(model_name)
    if not isinstance(model, Mapping):
        _issue(errors, "model_unknown", "deployment", "#/axes/model", f"unknown model {model_name!r}")
    else:
        revision, _revision_source, revision_conflict = _resolved_model_revision(
            fixed, model_name, bindings
        )
        if revision_conflict:
            _issue(
                errors,
                "model_revision_conflict",
                "deployment",
                f"#/runtime_bindings/model_revisions/{model_name}",
                "runtime model revision differs from the frozen JSON revision",
            )
        if not _REVISION_RE.fullmatch(revision):
            _issue(
                errors,
                "model_revision_missing",
                "deployment",
                f"#/fixed/provenance/model_revisions/revisions/{model_name}",
                "an immutable 40- or 64-hex model snapshot revision is required",
            )

    if not _REVISION_RE.fullmatch(str(bindings.simulator_revision)):
        _issue(
            errors,
            "simulator_revision_missing",
            "deployment",
            "#/runtime_bindings/simulator_revision",
            "the runner must bind its immutable implementation content SHA-256",
        )
    hardware = _resolved_hardware_timing(fixed, bindings)
    timing_class = str(hardware["timing_class"])
    if timing_class not in {"calibrated", "modeled"}:
        _issue(
            errors,
            "hardware_timing_provenance_missing",
            "deployment",
            "#/runtime_bindings/hardware_timing_class",
            "choose calibrated or explicitly modeled hardware timing",
        )
    elif timing_class == "modeled" and not bool(hardware["modeled_authorized"]):
        _issue(
            errors,
            "modeled_hardware_not_authorized",
            "deployment",
            "#/runtime_bindings/allow_modeled_hardware",
            "modeled timing must be explicitly authorized and remains non-calibrated",
        )
    if timing_class in {"calibrated", "modeled"}:
        if not str(hardware["source"]):
            _issue(
                errors,
                "hardware_timing_source_missing",
                "deployment",
                "#/runtime_bindings/hardware_timing_source",
                "hardware timing source identity is required",
            )
        if not _SHA256_RE.fullmatch(str(hardware["sha256"])):
            _issue(
                errors,
                "hardware_timing_hash_missing",
                "deployment",
                "#/runtime_bindings/hardware_timing_sha256",
                "hardware timing source SHA-256 is required",
            )

    policy_name = str(_selected(resolved_deployment, axes, "policy", fixed["aegiskv"]["central_policy"]))
    if policy_name not in {
        "recompute",
        "no_protection_control",
        "host_checkpoint",
        "neighbor_replica",
        "aegiskv",
    }:
        _issue(errors, "policy_unsupported", "deployment", "#/axes/policy", f"unsupported policy {policy_name!r}")
    algorithm = str(_selected(resolved_deployment, axes, "algorithm_variant", "full"))
    if algorithm == "no_algorithm_1":
        variant = fixed["aegiskv"].get("algorithm_variants", {}).get(
            "no_algorithm_1", {}
        )
        if not (
            variant.get("runtime_adapter")
            == "aegis_l1_l2.disable_slack_ordering"
            and variant.get("runtime_value") is True
            and float(variant.get("ordering_slack_ms", -1)) == 0.0
        ):
            _issue(
                errors,
                "slack_ordering_disable_adapter_missing",
                "deployment",
                "#/fixed/aegiskv/algorithm_variants/no_algorithm_1",
                "no_algorithm_1 requires the verified zero-slack runtime adapter",
            )
    return RuntimeReadiness(tuple(errors))


def _compile_transfer_catalog(
    fixed: Mapping[str, Any], topology: OfflinePhysicalTopology
) -> TransferResourceCatalog:
    resources: set[TransferResourceId] = set()
    engines = fixed["copy_resources"]["gpu_copy_engines"]
    for spec in topology.gpus:
        gpu = spec.gpu
        for lane in range(int(engines["d2d_per_gpu"])):
            resources.add(TransferResourceId("gpu-d2d-read", gpu.key, lane))
            resources.add(TransferResourceId("gpu-d2d-write", gpu.key, lane))
        for lane in range(int(engines["d2h_per_gpu"])):
            resources.add(TransferResourceId("gpu-d2h-read", gpu.key, lane))
        for lane in range(int(engines["h2d_per_gpu"])):
            resources.add(TransferResourceId("gpu-h2d-write", gpu.key, lane))
    dram = fixed["copy_resources"]["dram_lanes_per_node"]
    for node in topology.nodes:
        resources.update(
            {
                TransferResourceId("nvlink", node.node_id),
                TransferResourceId("pcie-egress", node.node_id),
                TransferResourceId("pcie-ingress", node.node_id),
                TransferResourceId("pcie-d2h", node.node_id),
                TransferResourceId("pcie-h2d", node.node_id),
                TransferResourceId("nic-egress", node.node_id),
                TransferResourceId("nic-ingress", node.node_id),
            }
        )
        for lane in range(int(dram["read"])):
            resources.add(TransferResourceId("dram-read", node.node_id, lane))
        for lane in range(int(dram["write"])):
            resources.add(TransferResourceId("dram-write", node.node_id, lane))
    resources.add(TransferResourceId("fabric", "cluster-fabric-0"))
    paths = {
        str(name): tuple(map(str, values))
        for name, values in fixed["copy_resources"]["path_resources"].items()
    }
    # The same catalogue covers protection copies and the physical recovery
    # loads emitted by OfflineAegisTransferBridge.plan_failover_recovery.
    profiles: dict[str, Mapping[str, int | str]] = {}
    for name, profile in fixed["copy_resources"]["links"].items():
        profiles[name] = MappingProxyType(
            {
                "peak_bandwidth_bytes_per_s": int(profile["peak_bandwidth_bytes_per_s"]),
                "setup_ns": int(profile["setup_ns"]),
                "contention": str(profile["contention"]),
            }
        )
    profiles["dram"] = MappingProxyType(
        {
            "peak_bandwidth_bytes_per_s": int(dram["bandwidth_bytes_per_s"]),
            "setup_ns": int(dram["setup_ns"]),
            "contention": str(dram["contention"]),
        }
    )
    return TransferResourceCatalog(
        resources=tuple(sorted(resources)),
        path_contracts=MappingProxyType(paths),
        profiles=MappingProxyType(profiles),
    )


def _copy_fabric(
    fixed: Mapping[str, Any], axes: Mapping[str, Any], resolved: Mapping[str, Any]
) -> CopyFabricConfig:
    links = fixed["copy_resources"]["links"]
    dram = fixed["copy_resources"]["dram_lanes_per_node"]
    selected_gb_s = float(
        _selected(
            resolved,
            axes,
            "cross_node_bandwidth_gb_s",
            int(links["cross_node_fabric"]["central_effective_bandwidth_bytes_per_s"]) / GB,
        )
    )
    pcie_setup = int(links["pcie"]["setup_ns"])
    nic_setup = int(links["nic_rdma"]["setup_ns"])
    fabric_setup = int(links["cross_node_fabric"]["setup_ns"])
    dram_setup = int(dram["setup_ns"])
    return CopyFabricConfig(
        bandwidths=CopyFabricBandwidths(
            gpu_copy_bytes_per_s=int(links["nvlink_nvswitch"]["peak_bandwidth_bytes_per_s"]),
            nvlink_bytes_per_s=int(links["nvlink_nvswitch"]["peak_bandwidth_bytes_per_s"]),
            pcie_bytes_per_s=int(links["pcie"]["peak_bandwidth_bytes_per_s"]),
            nic_bytes_per_s=int(links["nic_rdma"]["peak_bandwidth_bytes_per_s"]),
            fabric_bytes_per_s=round(selected_gb_s * GB),
            dram_write_bytes_per_s=int(dram["bandwidth_bytes_per_s"]),
            dram_read_bytes_per_s=int(dram["bandwidth_bytes_per_s"]),
        ),
        setup=CopyFabricSetupLatencies(
            same_node_d2d_ns=int(links["nvlink_nvswitch"]["setup_ns"]),
            cross_node_d2d_ns=2 * pcie_setup + 2 * nic_setup + fabric_setup,
            same_node_d2h_ns=pcie_setup + dram_setup,
            cross_node_d2h_ns=2 * pcie_setup + 2 * nic_setup + fabric_setup + dram_setup,
            same_node_h2d_ns=dram_setup + pcie_setup,
            cross_node_h2d_ns=dram_setup + 2 * pcie_setup + 2 * nic_setup + fabric_setup,
        ),
        fabric_id="cluster-fabric-0",
    )


def compile_deployment(
    resolved_deployment: Mapping[str, Any],
    axes: Mapping[str, Any],
    bindings: RuntimeBindings,
) -> CompiledDeployment:
    readiness = deployment_layer_readiness(resolved_deployment, axes, bindings)
    if not readiness.ready:
        raise RuntimeNotReadyError(readiness.errors)
    fixed = resolved_deployment["fixed"]
    group = str(resolved_deployment["group"])
    model_name = str(_selected(resolved_deployment, axes, "model", fixed["central_model"]))
    model = fixed["models"][model_name]
    model_revision, model_revision_source, _conflict = _resolved_model_revision(
        fixed, model_name, bindings
    )
    hardware = _resolved_hardware_timing(fixed, bindings)
    node_count = int(
        _selected(
            resolved_deployment,
            axes,
            "node_count",
            fixed["cluster"]["central_node_count"],
        )
    )
    topology = build_offline_physical_topology(
        fixed, model_name=model_name, node_count=node_count
    )
    dp_size = int(model["dp_size_by_node_count"][str(node_count)])
    pp_size = int(model["pp_size"])
    tp_size = int(model["tp_size"])
    policy = _policy_contract(fixed, axes, resolved_deployment)
    hbm_budget_gib = int(
        _selected(
            resolved_deployment,
            axes,
            "hbm_budget_gib_per_gpu",
            fixed["gpu_hbm"]["central_aegis_budget_gib_per_gpu"],
        )
    )
    dram_budget_gib = int(
        _selected(
            resolved_deployment,
            axes,
            "dram_budget_gib_per_node",
            fixed["node_dram"]["central_aegis_budget_gib_per_node"],
        )
    )
    total_bpt = int(model["kv_bytes_per_token_total"])
    dram_quota_bytes = dram_budget_gib * GIB
    dram_quota_tokens, dram_quota_tail = divmod(
        dram_quota_bytes, total_bpt
    )
    shard_count = tp_size * pp_size
    if total_bpt % shard_count:
        raise RuntimeError("validated KV geometry became fractionally sharded")
    per_rank_bpt = total_bpt // shard_count
    page_size = 1
    l1_bytes = int(fixed["gpu_hbm"]["primary_kv_budget_bytes_per_gpu"]) + int(
        fixed["gpu_hbm"]["radix_prefix_budget_bytes_per_gpu"]
    )
    l2_bytes_per_physical_gpu = int(
        fixed["node_dram"]["hicache_primary_budget_bytes_per_node"]
    ) // int(fixed["cluster"]["gpus_per_node"])
    topology_node_index = {node.node_id: index for index, node in enumerate(topology.nodes)}
    dp_nodes = [
        topology_node_index[topology.group(dp_rank, 0)[0].gpu.node_id]
        for dp_rank in range(dp_size)
    ]
    aegis = fixed["aegiskv"]
    aimd = aegis["aimd"]
    initial_alpha = float(aimd["initial_alpha"])
    min_alpha = float(aimd["minimum_alpha"])
    max_alpha = float(aimd["maximum_alpha"])
    if policy.algorithm_variant == "no_algorithm_3":
        min_alpha = initial_alpha
        max_alpha = initial_alpha
    overlay = {
        "model_path": str(model["model_id"]),
        "model_revision": model_revision,
        "platform": {
            "accelerator": {
                "name": "a100_sxm",
                "hbm_capacity_gb": int(fixed["cluster"]["gpu_hbm_capacity_bytes"]) / GIB,
            },
            "num_device_per_node": int(fixed["cluster"]["gpus_per_node"]),
        },
        "scheduler": {
            "dp_size": dp_size,
            "pp_size": pp_size,
            "tp_size": tp_size,
            "load_balance_method": "round_robin",
        },
        "cache": {
            "page_size": page_size,
            "l1_capacity_pages": l1_bytes // per_rank_bpt,
            "l2_capacity_pages": l2_bytes_per_physical_gpu // per_rank_bpt,
            "aegis_total_l1_capacity_pages": (hbm_budget_gib * GIB) // per_rank_bpt,
        },
        "communication": {
            "mode": "alpha_beta",
            "kv_bytes_per_token": per_rank_bpt,
        },
        "aegis_l1_l2": {
            "enabled": bool(policy.tiers),
            "committed_copy_protocol": True,
            "design_semantics": DESIGN_SEMANTICS,
            "prefix_chunk_tokens": int(aegis["prefix_chunk_tokens"]),
            "decode_chunk_tokens": int(aegis["sealed_decode_chunk_tokens"]),
            "host_alignment_tokens": 1,
            "require_explicit_per_node_dram_quota": True,
            "per_node_aegis_dram_quota_bytes": dram_quota_bytes,
            "per_node_aegis_dram_quota_tokens": dram_quota_tokens,
            "kv_bytes_per_logical_token": total_bpt,
            "disable_slack_ordering": (
                policy.algorithm_variant == "no_algorithm_1"
            ),
            "initial_alpha": initial_alpha,
            "min_alpha": min_alpha,
            "max_alpha": max_alpha,
            "aimd_increase_step": float(aimd["additive_increase"]),
            "aimd_decrease_factor": float(aimd["multiplicative_decrease"]),
            "topology": {
                "dp_rank_to_node": dp_nodes,
                "pp_stages_per_node": pp_size,
            },
        },
        "runtime_provenance": {
            "evidence_class": hardware["evidence_class"],
            "hardware_timing_source": hardware["source"],
            "hardware_timing_sha256": hardware["sha256"],
            "hardware_assumption_id": hardware["assumption_id"],
            "hardware_binding_source": hardware["binding_source"],
            "simulator_revision": bindings.simulator_revision,
        },
    }
    provenance = MappingProxyType(
        {
            "evidence_class": overlay["runtime_provenance"]["evidence_class"],
            "model_revision": model_revision,
            "model_revision_source": model_revision_source,
            "simulator_revision": bindings.simulator_revision,
            "simulator_revision_binding": (
                "runtime_implementation_content_sha256_v1"
            ),
            "hardware_timing_source": hardware["source"],
            "hardware_timing_sha256": hardware["sha256"],
            "hardware_assumption_id": hardware["assumption_id"],
            "hardware_binding_source": hardware["binding_source"],
            "resolved_deployment_sha256": config_hash(resolved_deployment),
            "copy_duration_rule": "sum_configured_setup_plus_bytes_over_bottleneck_v1",
        }
    )
    return CompiledDeployment(
        group=group,
        model_name=model_name,
        model_id=str(model["model_id"]),
        model_revision=model_revision,
        node_count=node_count,
        dp_size=dp_size,
        pp_size=pp_size,
        tp_size=tp_size,
        topology=topology,
        transfer_resources=_compile_transfer_catalog(fixed, topology),
        copy_fabric=_copy_fabric(fixed, axes, resolved_deployment),
        policy=policy,
        hbm_aegis_budget_bytes_per_gpu=hbm_budget_gib * GIB,
        dram_aegis_budget_bytes_per_node=dram_quota_bytes,
        dram_aegis_budget_tokens_per_node=dram_quota_tokens,
        dram_aegis_unallocatable_tail_bytes=dram_quota_tail,
        simulator_config_overlay=MappingProxyType(overlay),
        provenance=provenance,
    )


def _failure_scenario(
    resolved_failure: Mapping[str, Any], axes: Mapping[str, Any]
) -> tuple[bool, str]:
    fixed = resolved_failure["fixed"]
    fault = fixed["fault_model"]
    candidates = (
        "failure_scenario",
        "fault_domain",
        "failure_severity",
    )
    scenario = next(
        (
            str(axes[name])
            for name in candidates
            if name in resolved_failure.get("axes", {}) and name in axes
        ),
        str(fault.get("central_target") or "none"),
    )
    enabled = bool(fault["enabled"]) and scenario not in {"none", "no_failure"}
    return enabled, scenario


def _active_exposure_offset_ns(
    resolved_failure: Mapping[str, Any],
    deployment: CompiledDeployment | _FailureDeploymentShape,
) -> tuple[int, Mapping[str, object]]:
    trigger = resolved_failure["fixed"]["trigger"]
    normalization = trigger.get("scale_normalization")
    if normalization is None:
        value = int(trigger["active_offset_ns"])
        return value, MappingProxyType(
            {
                "rule": "fixed_active_offset_v1",
                "active_offset_ns": value,
            }
        )
    reference_nodes = int(normalization["reference_node_count"])
    reference_gpus_per_node = int(
        normalization["reference_gpus_per_node"]
    )
    reference_gpus = reference_nodes * reference_gpus_per_node
    reference_offset = int(normalization["reference_active_offset_ns"])
    selected_gpus = int(deployment.gpu_count)
    numerator = reference_offset * reference_gpus
    value = (numerator + selected_gpus // 2) // selected_gpus
    return value, MappingProxyType(
        {
            "rule": normalization["rule"],
            "reference_gpu_count": reference_gpus,
            "reference_active_offset_ns": reference_offset,
            "selected_gpu_count": selected_gpus,
            "conditioned_cluster_gpu_time_exposure_ns": selected_gpus * value,
            "active_offset_ns": value,
            "exposure_window": normalization["exposure_window"],
            "rounding": normalization["rounding"],
        }
    )


def _validated_active_exposure_plan(
    plan: Mapping[str, Any] | None,
    *,
    scenario: str,
    deployment: CompiledDeployment,
    expected_offset_ns: int,
) -> dict[str, object]:
    if not isinstance(plan, Mapping):
        raise RuntimeNotReadyError(
            (
                RuntimeReadinessError(
                    "active_exposure_plan_required",
                    "failure",
                    "#/runtime/active_exposure_plan",
                    "formal active failures require a pre-execution shadow plan",
                ),
            )
        )
    required = {
        "schema",
        "construction",
        "request_universe_sha256",
        "executed_request_set_sha256",
        "seed",
        "scenario",
        "physical_topology_sha256",
        "anchor_selector",
        "anchor_request_id",
        "anchor_arrival_ns",
        "anchor_source_dp_rank",
        "anchor_source_node_id",
        "target_selector",
        "planned_targets",
        "active_offset_rule",
        "active_offset_ns",
        "policy_inputs_excluded",
        "shadow_plan_sha256",
    }
    if set(plan) != required:
        raise RuntimeNotReadyError(
            (
                RuntimeReadinessError(
                    "active_exposure_plan_fields_invalid",
                    "failure",
                    "#/runtime/active_exposure_plan",
                    "active-exposure plan fields differ from the canonical schema",
                ),
            )
        )
    value = copy.deepcopy(dict(plan))
    supplied_hash = str(value.pop("shadow_plan_sha256") or "")
    expected_hash = config_hash(value)
    problems: list[str] = []
    if plan["schema"] != ACTIVE_EXPOSURE_PLAN_SCHEMA:
        problems.append("schema")
    if plan["construction"] != "pre_execution_request_universe_projection_v1":
        problems.append("construction")
    if not _SHA256_RE.fullmatch(str(plan["request_universe_sha256"])):
        problems.append("request_universe_sha256")
    if not _SHA256_RE.fullmatch(str(plan["executed_request_set_sha256"])):
        problems.append("executed_request_set_sha256")
    if not _SHA256_RE.fullmatch(str(plan["physical_topology_sha256"])):
        problems.append("physical_topology_sha256")
    if str(plan["physical_topology_sha256"]) != config_hash(
        deployment.topology.as_dict()
    ):
        problems.append("physical_topology_mismatch")
    if str(plan["scenario"]) != str(scenario):
        problems.append("scenario")
    if not str(plan["anchor_request_id"]):
        problems.append("anchor_request_id")
    try:
        source_rank = int(plan["anchor_source_dp_rank"])
        anchor_arrival_ns = int(plan["anchor_arrival_ns"])
        active_offset_ns = int(plan["active_offset_ns"])
    except (TypeError, ValueError):
        source_rank = -1
        anchor_arrival_ns = -1
        active_offset_ns = -1
        problems.append("integer_fields")
    if not 0 <= source_rank < deployment.dp_size:
        problems.append("anchor_source_dp_rank")
    if anchor_arrival_ns < 0:
        problems.append("anchor_arrival_ns")
    if active_offset_ns != int(expected_offset_ns):
        problems.append("active_offset_ns")
    if str(plan["anchor_selector"]) != "minimum_arrival_ns_then_request_id_v1":
        problems.append("anchor_selector")
    excluded = set(plan["policy_inputs_excluded"] or ())
    if not {
        "deployment.policy",
        "runtime_completion_time",
        "runtime_failure_outcome",
    } <= excluded:
        problems.append("policy_inputs_excluded")
    if not isinstance(plan["planned_targets"], list) or not plan["planned_targets"]:
        problems.append("planned_targets")
    if supplied_hash != expected_hash:
        problems.append("shadow_plan_sha256")
    if problems:
        raise RuntimeNotReadyError(
            (
                RuntimeReadinessError(
                    "active_exposure_plan_invalid",
                    "failure",
                    "#/runtime/active_exposure_plan",
                    "invalid plan fields: " + ", ".join(sorted(set(problems))),
                ),
            )
        )
    return copy.deepcopy(dict(plan))


def _failure_trigger_ns(
    resolved_failure: Mapping[str, Any],
    deployment: CompiledDeployment | _FailureDeploymentShape,
    *,
    anchor_active_at_ns: int | None = None,
) -> tuple[int, Mapping[str, object]]:
    trigger = resolved_failure["fixed"]["trigger"]
    if trigger.get("mode") == "policy_independent_active_exposure_v1":
        offset_ns, offset_metadata = _active_exposure_offset_ns(
            resolved_failure, deployment
        )
        if anchor_active_at_ns is None or int(anchor_active_at_ns) < 0:
            raise RuntimeNotReadyError(
                (
                    RuntimeReadinessError(
                        "anchor_activation_required",
                        "failure",
                        "#/runtime/anchor_active_at_ns",
                        "active-exposure failure must bind to causal Scheduler ownership",
                    ),
                )
            )
        value = int(anchor_active_at_ns) + offset_ns
        return value, MappingProxyType(
            {
                **dict(offset_metadata),
                "trigger_basis": "anchor_scheduler_ownership_relative_v1",
                "anchor_active_at_ns": int(anchor_active_at_ns),
                "trigger_time_ns": value,
            }
        )
    value = int(trigger["central_time_ns"])
    return value, MappingProxyType(
        {"rule": "absolute_deterministic_time_v1", "trigger_time_ns": value}
    )


def failure_layer_readiness(
    resolved_failure: Mapping[str, Any],
    axes: Mapping[str, Any],
    deployment: CompiledDeployment | _FailureDeploymentShape | None,
    bindings: RuntimeBindings | None = None,
) -> RuntimeReadiness:
    bindings = bindings or RuntimeBindings()
    errors = _axis_errors(resolved_failure, axes, layer="failure")
    group = _resolved_group(resolved_failure, "failure")
    if not group:
        _issue(errors, "layer_mismatch", "failure", "#/layer", "resolved failure v3 layer required")
        return RuntimeReadiness(tuple(errors))
    fixed = resolved_failure.get("fixed")
    if not isinstance(fixed, Mapping):
        _issue(errors, "fixed_missing", "failure", "#/fixed", "resolved fixed object required")
        return RuntimeReadiness(tuple(errors))
    enabled, scenario = _failure_scenario(resolved_failure, axes)
    if not enabled:
        return RuntimeReadiness(tuple(errors))
    if deployment is None:
        _issue(
            errors,
            "deployment_compilation_required",
            "failure",
            "#/runtime/deployment",
            "failure target resolution requires compiled physical topology",
        )
        return RuntimeReadiness(tuple(errors))
    match = re.fullmatch(r"(worker|gpu|process|node)_(\d+)", scenario)
    if match is None or int(match.group(2)) <= 0:
        _issue(errors, "failure_scenario_unsupported", "failure", "#/axes", f"unsupported failure scenario {scenario!r}")
    else:
        scope, count_text = match.groups()
        count = int(count_text)
        capacity = deployment.node_count if scope == "node" else deployment.dp_size
        if count >= capacity:
            _issue(
                errors,
                "failure_removes_all_capacity",
                "failure",
                "#/axes",
                f"{scenario} leaves no healthy {'node' if scope == 'node' else 'DP rank'}",
            )
        if scope == "gpu" and deployment.tp_size > 1:
            _issue(
                errors,
                "single_gpu_fail_stop_not_supported_for_tp",
                "failure",
                "#/axes",
                "current OfflineCluster hard GPU API addresses one DP rank and would incorrectly erase every TP shard",
            )
        if scope == "gpu" and deployment.pp_size > 1:
            _issue(
                errors,
                "single_gpu_fail_stop_not_supported_for_pp",
                "failure",
                "#/axes",
                "current OfflineCluster hard GPU API would incorrectly restart every PP stage in the DP replica",
            )
        if scope == "process" and deployment.pp_size > 1:
            _issue(
                errors,
                "single_process_fail_stop_not_supported_for_pp",
                "failure",
                "#/axes",
                "current OfflineCluster hard process API addresses one DP rank and would incorrectly restart every PP process",
            )
    if bindings.failure_target_selector not in {
        "dense_first_v1",
        "seed_mod_dense_order_v1",
        "active_anchor_plan_v1",
    }:
        _issue(
            errors,
            "failure_target_selector_missing",
            "failure",
            "#/runtime_bindings/failure_target_selector",
            "choose an explicit deterministic physical target selector",
        )
    elif (
        isinstance(deployment, CompiledDeployment)
        and match is not None
        and int(match.group(2)) > 0
    ):
        scope, count_text = match.groups()
        count = int(count_text)
        capacity = deployment.node_count if scope == "node" else deployment.dp_size
        if count < capacity:
            try:
                starts = (
                    range(capacity)
                    if bindings.failure_target_selector
                    == "active_anchor_plan_v1"
                    else (
                        _target_offset(
                            axes, capacity, bindings.failure_target_selector
                        ),
                    )
                )
                for start in starts:
                    for ordinal in range(count):
                        target_index = (start + ordinal) % capacity
                        resolved_target = resolve_offline_failure_target(
                            deployment.topology,
                            failure_scope=scope,
                            dp_rank=None if scope == "node" else target_index,
                            node_id=(
                                f"node{target_index}" if scope == "node" else None
                            ),
                        )
                        assert_current_cluster_representable(
                            deployment.topology, resolved_target
                        )
            except RuntimeNotReadyError as exc:
                errors.extend(exc.errors)
            except (FailureTargetNotRepresentableError, ValueError) as exc:
                _issue(
                    errors,
                    "failure_target_not_representable",
                    "failure",
                    "#/runtime/resolved_physical_target",
                    str(exc),
                )
    trigger = fixed["trigger"]
    normalization = trigger.get("scale_normalization")
    if normalization is not None:
        valid = isinstance(normalization, Mapping) and (
            normalization.get("rule")
            == "constant_conditioned_cluster_gpu_time_exposure_v2"
            and normalization.get("selected_gpu_count_source")
            == "compiled_physical_topology"
            and normalization.get("trigger_offset_formula")
            == "reference_active_offset_ns * reference_gpu_count / selected_gpu_count"
            and normalization.get("exposure_window")
            == "after_anchor_scheduler_ownership_before_injection"
            and normalization.get("rounding")
            == "nearest_integer_ns_half_up"
        )
        try:
            positive = (
                int(normalization["reference_node_count"]) > 0
                and int(normalization["reference_gpus_per_node"]) > 0
                and int(normalization["reference_active_offset_ns"]) > 0
                and deployment.gpu_count > 0
            )
        except (KeyError, TypeError, ValueError):
            positive = False
        if not valid or not positive:
            _issue(
                errors,
                "failure_scale_normalization_formula_invalid",
                "failure",
                "#/fixed/trigger/scale_normalization",
                "P3 requires the explicit conditioned cluster GPU-time exposure formula",
            )
    repair = fixed["repair"]
    nonzero_post_repair = (
        float(repair["replacement_route_delay_ms"]) > 0
        or float(repair["routing_outage_ms"]) > 0
    )
    if nonzero_post_repair and bindings.repair_delay_policy not in {
        "serial_sum_v1",
        "overlap_max_v1",
    }:
        _issue(
            errors,
            "repair_delay_policy_missing",
            "failure",
            "#/runtime_bindings/repair_delay_policy",
            "declare whether replacement-route and routing-outage delays sum or overlap",
        )
    return RuntimeReadiness(tuple(errors))


def _repair_duration_ns(
    resolved_failure: Mapping[str, Any], axes: Mapping[str, Any], bindings: RuntimeBindings
) -> int:
    repair = resolved_failure["fixed"]["repair"]
    horizon_s = float(
        _selected(
            resolved_failure,
            axes,
            "repair_horizon_s",
            repair["central_horizon_s"],
        )
    )
    replacement_s = float(repair["replacement_route_delay_ms"]) / 1000.0
    routing_s = float(repair["routing_outage_ms"]) / 1000.0
    if bindings.repair_delay_policy == "serial_sum_v1":
        total_s = horizon_s + replacement_s + routing_s
    elif bindings.repair_delay_policy == "overlap_max_v1":
        total_s = horizon_s + max(replacement_s, routing_s)
    else:
        # No-failure timelines never reach this helper.
        raise RuntimeError("validated repair-delay policy became unavailable")
    duration_ns = round(total_s * 1_000_000_000)
    if duration_ns <= 0:
        raise RuntimeError("hard fail-stop duration must be positive")
    return duration_ns


def _target_offset(axes: Mapping[str, Any], capacity: int, selector: str) -> int:
    if selector == "dense_first_v1":
        return 0
    seed = axes.get("seed")
    if type(seed) is not int:
        raise RuntimeNotReadyError(
            (
                RuntimeReadinessError(
                    "seed_missing_for_target_selector",
                    "failure",
                    "#/axes/seed",
                    "seed_mod_dense_order_v1 requires an integer seed",
                ),
            )
        )
    return int(seed) % capacity


def compile_failure_timeline(
    resolved_failure: Mapping[str, Any],
    axes: Mapping[str, Any],
    deployment: CompiledDeployment,
    bindings: RuntimeBindings,
    *,
    active_exposure_plan: Mapping[str, Any] | None = None,
    anchor_active_at_ns: int | None = None,
) -> CompiledFailureTimeline:
    readiness = failure_layer_readiness(
        resolved_failure, axes, deployment, bindings
    )
    if not readiness.ready:
        raise RuntimeNotReadyError(readiness.errors)
    group = str(resolved_failure["group"])
    enabled, scenario = _failure_scenario(resolved_failure, axes)
    if not enabled:
        return CompiledFailureTimeline(
            group=group,
            scenario="no_failure",
            events=(),
            metadata=MappingProxyType(
                {
                    "failure_mode": "disabled",
                    "repair_delay_policy": "not_applicable",
                    "soft_metadata_failure_forbidden": True,
                }
            ),
        )
    match = re.fullmatch(r"(worker|gpu|process|node)_(\d+)", scenario)
    assert match is not None
    scope, count_text = match.groups()
    count = int(count_text)
    active_mode = (
        resolved_failure["fixed"]["trigger"].get("mode")
        == "policy_independent_active_exposure_v1"
    )
    active_plan: dict[str, object] | None = None
    if active_mode:
        expected_offset_ns, _offset_metadata = _active_exposure_offset_ns(
            resolved_failure, deployment
        )
        active_plan = _validated_active_exposure_plan(
            active_exposure_plan,
            scenario=scenario,
            deployment=deployment,
            expected_offset_ns=expected_offset_ns,
        )
    trigger_ns, trigger_metadata = _failure_trigger_ns(
        resolved_failure,
        deployment,
        anchor_active_at_ns=anchor_active_at_ns,
    )
    duration_ns = _repair_duration_ns(resolved_failure, axes, bindings)
    capacity = deployment.node_count if scope == "node" else deployment.dp_size
    if active_plan is not None:
        anchor_source = int(active_plan["anchor_source_dp_rank"])
        if scope == "node":
            anchor_node = str(active_plan["anchor_source_node_id"])
            node_ids = [node.node_id for node in deployment.topology.nodes]
            try:
                start = node_ids.index(anchor_node)
            except ValueError as exc:
                raise RuntimeNotReadyError(
                    (
                        RuntimeReadinessError(
                            "anchor_source_node_unknown",
                            "failure",
                            "#/runtime/active_exposure_plan/anchor_source_node_id",
                            f"anchor source node {anchor_node!r} is absent",
                        ),
                    )
                ) from exc
        else:
            start = anchor_source
    else:
        start = _target_offset(axes, capacity, bindings.failure_target_selector)
    selected = tuple((start + index) % capacity for index in range(count))
    if active_plan is not None:
        expected_targets = (
            [
                {
                    "failure_scope": "node",
                    "node_id": deployment.topology.nodes[target].node_id,
                }
                for target in selected
            ]
            if scope == "node"
            else [
                {"failure_scope": scope, "dp_rank": target}
                for target in selected
            ]
        )
        if active_plan["planned_targets"] != expected_targets:
            raise RuntimeNotReadyError(
                (
                    RuntimeReadinessError(
                        "active_exposure_target_mismatch",
                        "failure",
                        "#/runtime/active_exposure_plan/planned_targets",
                        "planned targets do not match the anchor source and scenario",
                    ),
                )
            )
    events: list[HardFailureEvent] = []
    for ordinal, target in enumerate(selected):
        failure_scope = "node" if scope == "node" else scope
        resolved_target = resolve_offline_failure_target(
            deployment.topology,
            failure_scope=failure_scope,
            dp_rank=None if scope == "node" else target,
            node_id=f"node{target}" if scope == "node" else None,
        )
        try:
            assert_current_cluster_representable(
                deployment.topology, resolved_target
            )
        except FailureTargetNotRepresentableError as exc:
            raise RuntimeNotReadyError(
                (
                    RuntimeReadinessError(
                        "failure_target_not_representable",
                        "failure",
                        "#/runtime/resolved_physical_target",
                        str(exc),
                    ),
                )
            ) from exc
        events.append(
            HardFailureEvent(
                event_id=f"{group.lower()}-{scenario}-{ordinal:02d}",
                scenario=scenario,
                inject_at_ns=trigger_ns,
                repair_at_ns=trigger_ns + duration_ns,
                failure_scope=failure_scope,
                dp_rank=None if scope == "node" else target,
                node_id=f"node{target}" if scope == "node" else None,
                resolved_physical_target=MappingProxyType(
                    resolved_target.as_dict()
                ),
            )
        )
    return CompiledFailureTimeline(
        group=group,
        scenario=scenario,
        events=tuple(events),
        metadata=MappingProxyType(
            {
                "failure_mode": "hard_fail_stop",
                "target_selector": bindings.failure_target_selector,
                "repair_delay_policy": bindings.repair_delay_policy,
                "trigger_normalization": dict(trigger_metadata),
                "trigger_mode": resolved_failure["fixed"]["trigger"]["mode"],
                "active_exposure_plan": copy.deepcopy(active_plan),
                "shadow_plan_sha256": (
                    None
                    if active_plan is None
                    else active_plan["shadow_plan_sha256"]
                ),
                "anchor_request_id": (
                    None
                    if active_plan is None
                    else active_plan["anchor_request_id"]
                ),
                "anchor_source_dp_rank": (
                    None
                    if active_plan is None
                    else active_plan["anchor_source_dp_rank"]
                ),
                "anchor_source_node_id": (
                    None
                    if active_plan is None
                    else active_plan["anchor_source_node_id"]
                ),
                "route_fence_before_recovery": True,
                "volatile_hbm_lost": True,
                "node_failure_loses_local_gpu_and_dram": True,
                "repair_restores_empty_volatile_state": True,
                "soft_metadata_failure_forbidden": True,
            }
        ),
    )


def cell_runtime_readiness(
    group: GroupConfig,
    axes: Mapping[str, Any],
    bindings: RuntimeBindings | None = None,
) -> RuntimeReadiness:
    """Audit deployment, hard-failure compilation, and known runtime seams."""

    bindings = bindings or RuntimeBindings()
    deployment_report = deployment_layer_readiness(
        group.deployment.resolved, axes, bindings
    )
    errors = list(deployment_report.errors)
    compiled: CompiledDeployment | None = None
    if deployment_report.ready:
        compiled = compile_deployment(group.deployment.resolved, axes, bindings)
    failure_shape: CompiledDeployment | _FailureDeploymentShape
    if compiled is not None:
        failure_shape = compiled
    else:
        deployment_fixed = group.deployment.fixed
        model_name = str(
            _selected(
                group.deployment.resolved,
                axes,
                "model",
                deployment_fixed["central_model"],
            )
        )
        node_count = int(
            _selected(
                group.deployment.resolved,
                axes,
                "node_count",
                deployment_fixed["cluster"]["central_node_count"],
            )
        )
        failure_shape = _FailureDeploymentShape(
            node_count=node_count,
            dp_size=int(
                deployment_fixed["models"][model_name]["dp_size_by_node_count"][
                    str(node_count)
                ]
            ),
            pp_size=int(deployment_fixed["models"][model_name]["pp_size"]),
            tp_size=int(deployment_fixed["models"][model_name]["tp_size"]),
            gpu_count=(
                node_count * int(deployment_fixed["cluster"]["gpus_per_node"])
            ),
        )
    errors.extend(
        failure_layer_readiness(
            group.failure.resolved,
            axes,
            failure_shape,
            bindings,
        ).errors
    )

    return RuntimeReadiness(tuple(errors))


@dataclass(frozen=True)
class GroupExecutionAudit:
    group: str
    planned_cells: int
    executable_cells: int
    blocked_cells: int
    error_cell_counts: Mapping[str, int]
    sample_errors: Mapping[str, RuntimeReadinessError]

    def as_dict(self) -> dict[str, object]:
        return {
            "group": self.group,
            "planned_cells": self.planned_cells,
            "executable_cells": self.executable_cells,
            "blocked_cells": self.blocked_cells,
            "error_cell_counts": dict(sorted(self.error_cell_counts.items())),
            "sample_errors": {
                code: error.as_dict()
                for code, error in sorted(self.sample_errors.items())
            },
        }


@dataclass(frozen=True)
class SuiteExecutionAudit:
    groups: tuple[GroupExecutionAudit, ...]

    @property
    def planned_cells(self) -> int:
        return sum(group.planned_cells for group in self.groups)

    @property
    def executable_cells(self) -> int:
        return sum(group.executable_cells for group in self.groups)

    @property
    def blocked_cells(self) -> int:
        return sum(group.blocked_cells for group in self.groups)

    @property
    def ready(self) -> bool:
        return self.blocked_cells == 0

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": COMPILER_SCHEMA,
            "scope": "P1-P12",
            "ready": self.ready,
            "planned_cells": self.planned_cells,
            "executable_cells": self.executable_cells,
            "blocked_cells": self.blocked_cells,
            "groups": [group.as_dict() for group in self.groups],
        }


def audit_p1_p12_execution(
    suite: SuiteConfig, bindings: RuntimeBindings | None = None
) -> SuiteExecutionAudit:
    """Mechanically audit every one of the 2,134 P1-P12 cells."""

    bindings = bindings or RuntimeBindings()
    audits: list[GroupExecutionAudit] = []
    for group in suite.groups:
        if group.group not in SUPPORTED_GROUPS:
            continue
        planned = 0
        executable = 0
        counts: dict[str, int] = {}
        samples: dict[str, RuntimeReadinessError] = {}
        for cell in group.iter_cells():
            planned += 1
            report = cell_runtime_readiness(group, cell["axes"], bindings)
            if report.ready:
                executable += 1
            seen_codes: set[str] = set()
            for error in report.errors:
                samples.setdefault(error.code, error)
                if error.code not in seen_codes:
                    counts[error.code] = counts.get(error.code, 0) + 1
                    seen_codes.add(error.code)
        audits.append(
            GroupExecutionAudit(
                group=group.group,
                planned_cells=planned,
                executable_cells=executable,
                blocked_cells=planned - executable,
                error_cell_counts=MappingProxyType(counts),
                sample_errors=MappingProxyType(samples),
            )
        )
    return SuiteExecutionAudit(tuple(audits))


__all__ = [
    "COMPILER_SCHEMA",
    "CompiledDeployment",
    "CompiledFailureTimeline",
    "GroupExecutionAudit",
    "HardFailureEvent",
    "ProtectionPolicyContract",
    "RuntimeBindings",
    "RuntimeNotReadyError",
    "RuntimeReadiness",
    "RuntimeReadinessError",
    "SuiteExecutionAudit",
    "TransferResourceCatalog",
    "audit_p1_p12_execution",
    "cell_runtime_readiness",
    "compile_deployment",
    "compile_failure_timeline",
    "deployment_layer_readiness",
    "failure_layer_readiness",
]
