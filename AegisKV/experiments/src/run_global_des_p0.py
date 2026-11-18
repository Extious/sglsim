"""Execute the frozen twelve-cell P0 global-DES contract suite.

P0 is a simulator contract gate, not a performance experiment.  Every cell is
constructed from the three frozen v3 JSON layers, executes real
``GlobalDES`` events, and publishes the complete raw-evidence contract.  The
suite configuration remains ``planned_unexecuted``: this runner only records
that the twelve contract cells were exercised.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Mapping, Sequence

from sglsim.srt.managers.data_parallel_controller import DataParallelController
from sglsim.srt.server_args import PortArgs, ServerArgs
from sglsim.srt.simulation.aegis_independent_tiers import DESIGN_SEMANTICS
from sglsim.srt.simulation.global_des import (
    CommitGroupState,
    EventPriority,
    GlobalDES,
    HbmAccount,
    KvCopyState,
    KvLocation,
    PhysicalGpuId,
    StorageTier,
    TransferRecord,
    TransferResourceId,
    TransferState,
)
from sglsim.srt.simulation.global_des_artifacts import (
    EVIDENCE_CLASS,
    write_global_des_artifacts,
)
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.simulation.offline_aegis_transfer_bridge import (
    CopyFabricBandwidths,
    CopyFabricConfig,
    CopyFabricSetupLatencies,
    OfflineAegisTransferBridge,
)
from sglsim.srt.simulation.offline_resource_bridge import (
    NodeDramSpec,
    OfflinePhysicalTopology,
    PhysicalGpuSpec,
)
from sglsim.srt.utils.network import _QUEUES

from .global_des_config import (
    CONFIG_ROOT,
    PLANNED_STATUS,
    P0_SCENARIOS,
    REQUIRED_INVARIANTS,
    GroupConfig,
    config_hash,
    load_suite,
)


SUMMARY_SCHEMA = "aegiskv.global-des-p0-summary.v1"
AUDIT_DETAILS_SCHEMA = "aegiskv.global-des-p0-contract-audit.v1"
ENGINE = "sglsim_global_des_p0_contract"
GROUP = "P0"


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


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _event_content_hash(des: GlobalDES) -> str:
    """Hash only modeled event content; paths and run lineage are absent."""

    return _sha256(des.export_timeseries()["journal"])


def _implementation_hash(config_root: Path) -> str:
    """Hash the exact local implementation/config inputs used by P0."""

    repo_root = Path(__file__).resolve().parents[3]
    inputs = (
        Path(__file__).resolve(),
        repo_root / "python/sglsim/srt/managers/data_parallel_controller.py",
        repo_root / "python/sglsim/srt/managers/aegis_independent_tier_manager.py",
        repo_root / "python/sglsim/srt/simulation/aegis_independent_tiers.py",
        repo_root / "python/sglsim/srt/simulation/global_des.py",
        repo_root / "python/sglsim/srt/simulation/global_des_artifacts.py",
        repo_root / "python/sglsim/srt/simulation/offline_aegis_transfer_bridge.py",
        repo_root / "python/sglsim/srt/simulation/offline_resource_bridge.py",
        repo_root / "python/sglsim/srt/simulation/manager/config.py",
        repo_root / "python/sglsim/srt/server_args.py",
        repo_root / "python/sglsim/srt/utils/network.py",
        repo_root / "AegisKV/experiments/src/global_des_config.py",
        *sorted(Path(config_root).rglob("*.json")),
    )
    digest = hashlib.sha256()
    for path in inputs:
        try:
            relative = path.relative_to(repo_root).as_posix()
        except ValueError:
            relative = path.name
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass
class _ScenarioRun:
    scenario: str
    seed: int
    des: GlobalDES = field(default_factory=GlobalDES)
    rank_steps: list[dict[str, Any]] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    assertions: dict[str, dict[str, Any]] = field(default_factory=dict)
    witnesses: dict[str, Any] = field(default_factory=dict)

    def assert_contract(self, name: str, passed: bool, evidence: Any) -> None:
        self.assertions[str(name)] = {
            "passed": bool(passed),
            "evidence": copy.deepcopy(evidence),
        }


def _resolved_three_layers(
    group: GroupConfig, *, scenario: str, seed: int
) -> dict[str, Any]:
    request = copy.deepcopy(group.request.fixed)
    request["selected_axes"] = {
        "contract_scenario": scenario,
        "seed": int(seed),
    }
    return {
        "request": request,
        "failure": copy.deepcopy(group.failure.fixed),
        "deployment": copy.deepcopy(group.deployment.fixed),
    }


def _ensure_node(run: _ScenarioRun, deployment: Mapping[str, Any], node: str) -> None:
    if node in run.des.node_dram:
        return
    run.des.register_node_dram(
        node,
        capacity_bytes=int(
            deployment["cluster"]["host_dram_capacity_bytes_per_node"]
        ),
        runtime_bytes=int(
            deployment["node_dram"]["runtime_reserve_bytes_per_node"]
        ),
    )


def _add_gpu(
    run: _ScenarioRun,
    deployment: Mapping[str, Any],
    *,
    node: str,
    local_gpu_id: int,
    dp_rank: int,
    pp_rank: int = 0,
    tp_rank: int = 0,
    model_name: str = "qwen3-8b",
    weight_shards: int | None = None,
) -> PhysicalGpuId:
    _ensure_node(run, deployment, node)
    gpu = PhysicalGpuId(
        node_id=node,
        local_gpu_id=local_gpu_id,
        dp_rank=dp_rank,
        pp_rank=pp_rank,
        tp_rank=tp_rank,
    )
    model = deployment["models"][model_name]
    shards = (
        int(model["tp_size"]) * int(model["pp_size"])
        if weight_shards is None
        else int(weight_shards)
    )
    total_weights = int(model["weight_bytes_total"])
    if shards <= 0 or total_weights % shards:
        raise ValueError("contract topology requires integral weight shards")
    run.des.register_gpu(
        gpu,
        capacity_bytes=int(deployment["cluster"]["gpu_hbm_capacity_bytes"]),
        weights_bytes=total_weights // shards,
        runtime_bytes=int(
            deployment["gpu_hbm"]["runtime_reserve_bytes_per_gpu"]
        ),
    )
    run.rank_steps.append(
        {
            "schema": "aegiskv.p0.physical-rank-step.v1",
            "step_id": f"topology/{gpu.key}",
            "timestamp_ns": run.des.now_ns,
            "phase": "physical_gpu_registered",
            **gpu.as_dict(),
            "model": model_name,
            "logical_tp_group_size": int(model["tp_size"]),
            "weight_shards": shards,
        }
    )
    return gpu


def _chunk_bytes(
    deployment: Mapping[str, Any],
    model_name: str,
    *,
    shard_count: int,
) -> int:
    total = (
        int(deployment["models"][model_name]["kv_bytes_per_token_total"])
        * int(deployment["aegiskv"]["prefix_chunk_tokens"])
    )
    if total % shard_count:
        raise ValueError("contract KV chunk does not divide physical shards")
    return total // shard_count


def _register_source_extent(
    run: _ScenarioRun,
    deployment: Mapping[str, Any],
    *,
    extent_id: str,
    request_id: str,
    source_gpu: PhysicalGpuId,
    model_name: str,
    parallel_layout: str,
    pp_rank: int,
    tp_rank: int,
    shard_count: int,
) -> tuple[str, KvLocation]:
    size_bytes = _chunk_bytes(
        deployment, model_name, shard_count=shard_count
    )
    chunk_tokens = int(deployment["aegiskv"]["prefix_chunk_tokens"])
    run.des.register_extent(
        extent_id=extent_id,
        request_id=request_id,
        size_bytes=size_bytes,
        token_start=0,
        token_end=chunk_tokens,
        route_version=1,
        ordinal=0,
        model_revision=f"{deployment['models'][model_name]['model_id']}@p0-contract",
        kv_abi="sglsim-p0-bf16-kv-v1",
        parallel_layout=parallel_layout,
        pp_rank=pp_rank,
        tp_rank=tp_rank,
        shard_count=shard_count,
        source_failure_domain=source_gpu.node_id,
        workflow_id=f"workflow/{request_id}",
    )
    source = KvLocation.gpu_primary(source_gpu)
    run.des.place_extent(extent_id, source, reason="p0_source_primary")
    return extent_id, source


def _register_dram_source_extent(
    run: _ScenarioRun,
    deployment: Mapping[str, Any],
    *,
    extent_id: str,
    request_id: str,
    node: str,
    model_name: str = "qwen3-8b",
) -> tuple[str, KvLocation]:
    _ensure_node(run, deployment, node)
    chunk_tokens = int(deployment["aegiskv"]["prefix_chunk_tokens"])
    size_bytes = _chunk_bytes(deployment, model_name, shard_count=1)
    run.des.register_extent(
        extent_id=extent_id,
        request_id=request_id,
        size_bytes=size_bytes,
        token_start=0,
        token_end=chunk_tokens,
        model_revision=f"{deployment['models'][model_name]['model_id']}@p0-contract",
        kv_abi="sglsim-p0-bf16-kv-v1",
        parallel_layout="tp1-pp1",
        shard_count=1,
        source_failure_domain=node,
        workflow_id=f"workflow/{request_id}",
    )
    source = KvLocation.node_dram(node)
    run.des.place_extent(extent_id, source, reason="p0_dram_source")
    return extent_id, source


def _path_resources(
    deployment: Mapping[str, Any],
    path: str,
    *,
    source: KvLocation,
    destination: KvLocation,
) -> tuple[TransferResourceId, ...]:
    names = tuple(
        deployment["copy_resources"]["path_resources"][path]
    )
    source_node = source.gpu.node_id if source.gpu is not None else source.node_id
    destination_node = (
        destination.gpu.node_id
        if destination.gpu is not None
        else destination.node_id
    )
    assert source_node is not None and destination_node is not None
    owner_by_name = {
        "source_d2d_engine": source.gpu.key if source.gpu else source_node,
        "source_d2h_engine": source.gpu.key if source.gpu else source_node,
        "source_h2d_engine": source.gpu.key if source.gpu else source_node,
        "target_d2d_engine": (
            destination.gpu.key if destination.gpu else destination_node
        ),
        "target_h2d_engine": (
            destination.gpu.key if destination.gpu else destination_node
        ),
        "source_pcie": source_node,
        "target_pcie": destination_node,
        "source_nic": source_node,
        "target_nic": destination_node,
        "cross_node_fabric": f"{source_node}->{destination_node}",
        "nvlink_nvswitch": source_node,
        "destination_dram_write_lane": destination_node,
        "source_dram_read_lane": source_node,
    }
    return tuple(
        TransferResourceId(name, owner_by_name[name]) for name in names
    )


def _duration_ns(
    deployment: Mapping[str, Any], path: str, size_bytes: int
) -> int:
    copy_resources = deployment["copy_resources"]
    links = copy_resources["links"]
    dram = copy_resources["dram_lanes_per_node"]
    if path == "same_node_peer_hbm":
        bandwidth = int(links["nvlink_nvswitch"]["peak_bandwidth_bytes_per_s"])
        setup = int(links["nvlink_nvswitch"]["setup_ns"])
    elif path == "cross_node_peer_hbm":
        bandwidth = min(
            int(links["pcie"]["peak_bandwidth_bytes_per_s"]),
            int(links["nic_rdma"]["peak_bandwidth_bytes_per_s"]),
            int(
                links["cross_node_fabric"][
                    "central_effective_bandwidth_bytes_per_s"
                ]
            ),
        )
        setup = (
            2 * int(links["pcie"]["setup_ns"])
            + 2 * int(links["nic_rdma"]["setup_ns"])
            + int(links["cross_node_fabric"]["setup_ns"])
        )
    elif path in {"source_to_node_dram", "node_dram_to_recovery_gpu"}:
        bandwidth = min(
            int(links["pcie"]["peak_bandwidth_bytes_per_s"]),
            int(dram["bandwidth_bytes_per_s"]),
        )
        setup = int(links["pcie"]["setup_ns"]) + int(dram["setup_ns"])
    else:
        raise ValueError(f"unsupported frozen transfer path {path!r}")
    return setup + math.ceil(int(size_bytes) * 1_000_000_000 / bandwidth)


def _register_group(
    run: _ScenarioRun,
    *,
    group_id: str,
    tier: StorageTier,
    expected: Sequence[str],
    parallel_layout: str,
    model_revision: str,
) -> None:
    run.des.register_commit_group(
        group_id=group_id,
        tier=tier,
        route_version=1,
        model_revision=model_revision,
        kv_abi="sglsim-p0-bf16-kv-v1",
        parallel_layout=parallel_layout,
        expected_shard_keys=tuple(expected),
    )


def _reserve(
    run: _ScenarioRun,
    deployment: Mapping[str, Any],
    *,
    extent_id: str,
    source: KvLocation,
    destination: KvLocation,
    path: str,
    group_id: str | None = None,
    shard_key: str | None = None,
) -> TransferRecord:
    extent = run.des.extents[extent_id]
    return run.des.reserve_transfer(
        extent_id=extent_id,
        source=source,
        destination=destination,
        resources=_path_resources(
            deployment, path, source=source, destination=destination
        ),
        duration_ns=_duration_ns(deployment, path, extent.size_bytes),
        group_id=group_id,
        shard_key=shard_key,
    )


def _request_row(
    run: _ScenarioRun,
    deployment: Mapping[str, Any],
    *,
    request_id: str,
    model: str,
    dp_rank: int = 0,
) -> None:
    run.requests.append(
        {
            "schema": "aegiskv.p0.request.v1",
            "request_id": request_id,
            "arrival_ns": 0,
            "contract_scenario": run.scenario,
            "seed": run.seed,
            "model": model,
            "context_tokens": int(deployment["aegiskv"]["prefix_chunk_tokens"]),
            "dp_rank": dp_rank,
        }
    )


def _record_gpu_failure(
    run: _ScenarioRun,
    *,
    gpu: PhysicalGpuId,
    failure_id: str,
) -> Callable[[GlobalDES, Any], None]:
    def callback(des: GlobalDES, _event: Any) -> None:
        affected = des.fail_gpu(gpu)
        run.failures.append(
            {
                "schema": "aegiskv.p0.failure.v1",
                "failure_id": failure_id,
                "failure_ns": des.now_ns,
                "domain": "gpu",
                "target": gpu.key,
                "affected_extents": list(affected),
            }
        )

    return callback


def _record_node_failure(
    run: _ScenarioRun, *, node: str, failure_id: str
) -> Callable[[GlobalDES, Any], None]:
    def callback(des: GlobalDES, _event: Any) -> None:
        affected = des.fail_node(node)
        run.failures.append(
            {
                "schema": "aegiskv.p0.failure.v1",
                "failure_id": failure_id,
                "failure_ns": des.now_ns,
                "domain": "node",
                "target": node,
                "affected_extents": list(affected),
            }
        )

    return callback


def _scenario_independent_dp_overlap(
    run: _ScenarioRun, deployment: Mapping[str, Any]
) -> None:
    completions: list[tuple[int, int]] = []
    tick = int(deployment["aegiskv"]["sealed_decode_chunk_tokens"])
    for rank, multiplier in ((0, 1), (1, 2)):
        _add_gpu(
            run,
            deployment,
            node="node0",
            local_gpu_id=rank,
            dp_rank=rank,
        )
        request_id = f"p0-overlap-dp{rank}"
        _request_row(
            run, deployment, request_id=request_id, model="qwen3-8b", dp_rank=rank
        )
        duration = tick * multiplier

        def complete(
            des: GlobalDES, _event: Any, *, dp_rank: int = rank, dur: int = duration
        ) -> None:
            completions.append((dp_rank, des.now_ns))
            run.rank_steps.append(
                {
                    "schema": "aegiskv.p0.physical-rank-step.v1",
                    "step_id": f"decode-complete/dp{dp_rank}",
                    "timestamp_ns": des.now_ns,
                    "phase": "decode_complete",
                    "dp_rank": dp_rank,
                    "pp_rank": 0,
                    "tp_rank": 0,
                    "duration_ns": dur,
                }
            )

        run.des.schedule_after(
            duration,
            kind=f"dp{rank}_decode_complete",
            priority=EventPriority.COMPUTE_COMPLETION,
            payload={"dp_rank": rank},
            callback=complete,
        )
    run.des.run()
    observed = {rank: timestamp for rank, timestamp in completions}
    expected_max = max(tick, 2 * tick)
    run.witnesses["non_lockstep_dp_overlap"] = {
        "completion_ns": {str(rank): timestamp for rank, timestamp in observed.items()},
        "max_ns": expected_max,
        "sum_ns": 3 * tick,
    }
    run.assert_contract(
        "independent_dp_max_not_sum",
        run.des.now_ns == expected_max
        and run.des.now_ns != 3 * tick
        and observed == {0: tick, 1: 2 * tick},
        run.witnesses["non_lockstep_dp_overlap"],
    )


def _scenario_qwen_tp(
    run: _ScenarioRun,
    deployment: Mapping[str, Any],
    *,
    model_name: str,
) -> None:
    model = deployment["models"][model_name]
    tp_size = int(model["tp_size"])
    layout = f"tp{tp_size}-pp1"
    source_gpus = [
        _add_gpu(
            run,
            deployment,
            node="node0",
            local_gpu_id=rank,
            dp_rank=0,
            tp_rank=rank,
            model_name=model_name,
        )
        for rank in range(tp_size)
    ]
    target_gpus = [
        _add_gpu(
            run,
            deployment,
            node="node0",
            local_gpu_id=tp_size + rank,
            dp_rank=1,
            tp_rank=rank,
            model_name=model_name,
        )
        for rank in range(tp_size)
    ]
    request_id = f"p0-{model_name}-tp{tp_size}"
    _request_row(run, deployment, request_id=request_id, model=model_name)
    expected = tuple(f"tp{rank}" for rank in range(tp_size))
    model_revision = f"{model['model_id']}@p0-contract"
    group_id = f"{request_id}/hbm"
    _register_group(
        run,
        group_id=group_id,
        tier=StorageTier.GPU_AEGIS,
        expected=expected,
        parallel_layout=layout,
        model_revision=model_revision,
    )
    transfers: list[TransferRecord] = []
    for rank, shard_key in enumerate(expected):
        extent_id, source = _register_source_extent(
            run,
            deployment,
            extent_id=f"{request_id}/chunk0/{shard_key}",
            request_id=request_id,
            source_gpu=source_gpus[rank],
            model_name=model_name,
            parallel_layout=layout,
            pp_rank=0,
            tp_rank=rank,
            shard_count=tp_size,
        )
        transfers.append(
            _reserve(
                run,
                deployment,
                extent_id=extent_id,
                source=source,
                destination=KvLocation.gpu_aegis(target_gpus[rank]),
                path="same_node_peer_hbm",
                group_id=group_id,
                shard_key=shard_key,
            )
        )
    run.des.run()
    group = run.des.commit_groups[group_id]
    physical_tp = sorted(
        gpu.tp_rank for gpu in source_gpus if gpu.dp_rank == 0
    )
    run.witnesses["physical_tp_ranks"] = physical_tp
    run.assert_contract(
        f"{model_name}_physical_tp_shards",
        physical_tp == list(range(tp_size))
        and group.state is CommitGroupState.COMMITTED
        and all(row.state is TransferState.COMMITTED for row in transfers),
        {
            "model": model_name,
            "tp_size": tp_size,
            "physical_tp_ranks": physical_tp,
            "acked_shards": sorted(group.acked_shards),
            "group_state": group.state.value,
        },
    )


def _scenario_synthetic_pp2(
    run: _ScenarioRun, deployment: Mapping[str, Any]
) -> None:
    model_name = "qwen3-8b"
    layout = "tp1-pp2"
    source = [
        _add_gpu(
            run,
            deployment,
            node="node0",
            local_gpu_id=rank,
            dp_rank=0,
            pp_rank=rank,
            model_name=model_name,
            weight_shards=2,
        )
        for rank in range(2)
    ]
    targets = [
        _add_gpu(
            run,
            deployment,
            node="node0",
            local_gpu_id=2 + rank,
            dp_rank=1,
            pp_rank=rank,
            model_name=model_name,
            weight_shards=2,
        )
        for rank in range(2)
    ]
    request_id = "p0-synthetic-pp2"
    _request_row(run, deployment, request_id=request_id, model=model_name)
    expected = ("pp0/tp0", "pp1/tp0")
    group_id = f"{request_id}/hbm"
    model_revision = f"{deployment['models'][model_name]['model_id']}@p0-contract"
    _register_group(
        run,
        group_id=group_id,
        tier=StorageTier.GPU_AEGIS,
        expected=expected,
        parallel_layout=layout,
        model_revision=model_revision,
    )
    for pp_rank, shard_key in enumerate(expected):
        extent_id, location = _register_source_extent(
            run,
            deployment,
            extent_id=f"{request_id}/chunk0/{shard_key}",
            request_id=request_id,
            source_gpu=source[pp_rank],
            model_name=model_name,
            parallel_layout=layout,
            pp_rank=pp_rank,
            tp_rank=0,
            shard_count=2,
        )
        _reserve(
            run,
            deployment,
            extent_id=extent_id,
            source=location,
            destination=KvLocation.gpu_aegis(targets[pp_rank]),
            path="same_node_peer_hbm",
            group_id=group_id,
            shard_key=shard_key,
        )
    run.des.run()
    group = run.des.commit_groups[group_id]
    pp_ranks = sorted({gpu.pp_rank for gpu in source})
    run.witnesses["physical_pp_ranks"] = pp_ranks
    run.assert_contract(
        "synthetic_pp2_physical_shards",
        pp_ranks == [0, 1] and group.state is CommitGroupState.COMMITTED,
        {
            "parallel_layout": layout,
            "physical_pp_ranks": pp_ranks,
            "acked_shards": sorted(group.acked_shards),
            "group_state": group.state.value,
        },
    )


def _scenario_path(
    run: _ScenarioRun,
    deployment: Mapping[str, Any],
    *,
    path: str,
) -> None:
    request_id = f"p0-{path}"
    _request_row(run, deployment, request_id=request_id, model="qwen3-8b")
    if path == "node_dram_to_recovery_gpu":
        target_gpu = _add_gpu(
            run,
            deployment,
            node="node1",
            local_gpu_id=0,
            dp_rank=1,
        )
        extent_id, source = _register_dram_source_extent(
            run,
            deployment,
            extent_id=f"{request_id}/chunk0/tp0",
            request_id=request_id,
            node="node0",
        )
        destination = KvLocation.gpu_primary(target_gpu)
        transfer = _reserve(
            run,
            deployment,
            extent_id=extent_id,
            source=source,
            destination=destination,
            path=path,
        )
    else:
        source_gpu = _add_gpu(
            run,
            deployment,
            node="node0",
            local_gpu_id=0,
            dp_rank=0,
        )
        extent_id, source = _register_source_extent(
            run,
            deployment,
            extent_id=f"{request_id}/chunk0/tp0",
            request_id=request_id,
            source_gpu=source_gpu,
            model_name="qwen3-8b",
            parallel_layout="tp1-pp1",
            pp_rank=0,
            tp_rank=0,
            shard_count=1,
        )
        if path in {"same_node_peer_hbm", "cross_node_peer_hbm"}:
            target_node = "node0" if path == "same_node_peer_hbm" else "node1"
            target_gpu = _add_gpu(
                run,
                deployment,
                node=target_node,
                local_gpu_id=1 if target_node == "node0" else 0,
                dp_rank=1,
            )
            destination = KvLocation.gpu_aegis(target_gpu)
            tier = StorageTier.GPU_AEGIS
        elif path == "source_to_node_dram":
            destination = KvLocation.node_dram("node0")
            tier = StorageTier.NODE_DRAM
        else:  # Kept fail closed if the frozen dispatch table drifts.
            raise ValueError(f"unsupported P0 path {path!r}")
        group_id = f"{request_id}/{tier.value}"
        _register_group(
            run,
            group_id=group_id,
            tier=tier,
            expected=("tp0",),
            parallel_layout="tp1-pp1",
            model_revision="Qwen/Qwen3-8B@p0-contract",
        )
        transfer = _reserve(
            run,
            deployment,
            extent_id=extent_id,
            source=source,
            destination=destination,
            path=path,
            group_id=group_id,
            shard_key="tp0",
        )
    run.des.run()
    expected_names = list(
        deployment["copy_resources"]["path_resources"][path]
    )
    observed_names = [resource.kind for resource in transfer.resources]
    run.witnesses["transfer_path"] = {
        "path": path,
        "expected_resources": expected_names,
        "observed_resources": observed_names,
        "source_tier": transfer.source.tier.value,
        "destination_tier": transfer.destination.tier.value,
    }
    run.assert_contract(
        f"{path}_exact_resource_path",
        observed_names == expected_names
        and transfer.state is TransferState.COMMITTED,
        run.witnesses["transfer_path"],
    )


def _scenario_failure_in_pending(
    run: _ScenarioRun, deployment: Mapping[str, Any]
) -> None:
    source_gpu = _add_gpu(
        run, deployment, node="node0", local_gpu_id=0, dp_rank=0
    )
    target_gpu = _add_gpu(
        run, deployment, node="node1", local_gpu_id=0, dp_rank=1
    )
    request_id = "p0-failure-in-pending"
    _request_row(run, deployment, request_id=request_id, model="qwen3-8b")
    extent_id, source = _register_source_extent(
        run,
        deployment,
        extent_id=f"{request_id}/chunk0/tp0",
        request_id=request_id,
        source_gpu=source_gpu,
        model_name="qwen3-8b",
        parallel_layout="tp1-pp1",
        pp_rank=0,
        tp_rank=0,
        shard_count=1,
    )
    group_id = f"{request_id}/hbm"
    _register_group(
        run,
        group_id=group_id,
        tier=StorageTier.GPU_AEGIS,
        expected=("tp0",),
        parallel_layout="tp1-pp1",
        model_revision="Qwen/Qwen3-8B@p0-contract",
    )
    destination = KvLocation.gpu_aegis(target_gpu)
    transfer = _reserve(
        run,
        deployment,
        extent_id=extent_id,
        source=source,
        destination=destination,
        path="cross_node_peer_hbm",
        group_id=group_id,
        shard_key="tp0",
    )
    failure_ns = transfer.start_ns + max(1, (transfer.end_ns - transfer.start_ns) // 2)
    run.des.schedule_at(
        failure_ns,
        kind="failure_in_pending_window",
        priority=EventPriority.FAILURE,
        callback=_record_gpu_failure(
            run, gpu=source_gpu, failure_id="failure-in-pending-source"
        ),
    )
    run.des.run()
    group = run.des.commit_groups[group_id]
    run.witnesses["pending_failure"] = {
        "failure_ns": failure_ns,
        "transfer_start_ns": transfer.start_ns,
        "transfer_end_ns": transfer.end_ns,
        "transfer_state": transfer.state.value,
        "group_state": group.state.value,
        "destination_state": run.des.extents[extent_id].locations[
            destination
        ].value,
        "target_pending_bytes": run.des.gpu_hbm[target_gpu].aegis_pending,
        "target_committed_bytes": run.des.gpu_hbm[target_gpu].aegis_committed,
    }
    run.assert_contract(
        "pending_failure_aborts_and_releases",
        transfer.start_ns < failure_ns < transfer.end_ns
        and transfer.state is TransferState.ABORTED
        and group.state is CommitGroupState.ABORTED
        and run.des.extents[extent_id].locations[destination]
        is KvCopyState.ABORTED
        and run.des.gpu_hbm[target_gpu].aegis_pending == 0
        and run.des.gpu_hbm[target_gpu].aegis_committed == 0,
        run.witnesses["pending_failure"],
    )


def _scenario_failure_after_commit(
    run: _ScenarioRun, deployment: Mapping[str, Any]
) -> None:
    source_gpu = _add_gpu(
        run, deployment, node="node0", local_gpu_id=0, dp_rank=0
    )
    peer_gpu = _add_gpu(
        run, deployment, node="node1", local_gpu_id=0, dp_rank=1
    )
    request_id = "p0-failure-after-commit"
    _request_row(run, deployment, request_id=request_id, model="qwen3-8b")
    extent_id, source = _register_source_extent(
        run,
        deployment,
        extent_id=f"{request_id}/chunk0/tp0",
        request_id=request_id,
        source_gpu=source_gpu,
        model_name="qwen3-8b",
        parallel_layout="tp1-pp1",
        pp_rank=0,
        tp_rank=0,
        shard_count=1,
    )
    hbm_group = f"{request_id}/hbm"
    dram_group = f"{request_id}/dram"
    for group_id, tier in (
        (hbm_group, StorageTier.GPU_AEGIS),
        (dram_group, StorageTier.NODE_DRAM),
    ):
        _register_group(
            run,
            group_id=group_id,
            tier=tier,
            expected=("tp0",),
            parallel_layout="tp1-pp1",
            model_revision="Qwen/Qwen3-8B@p0-contract",
        )
    hbm_destination = KvLocation.gpu_aegis(peer_gpu)
    dram_destination = KvLocation.node_dram("node1")
    hbm_transfer = _reserve(
        run,
        deployment,
        extent_id=extent_id,
        source=source,
        destination=hbm_destination,
        path="cross_node_peer_hbm",
        group_id=hbm_group,
        shard_key="tp0",
    )
    dram_transfer = _reserve(
        run,
        deployment,
        extent_id=extent_id,
        source=source,
        destination=dram_destination,
        path="source_to_node_dram",
        group_id=dram_group,
        shard_key="tp0",
    )
    run.des.run()
    failure_ns = run.des.now_ns + 1
    run.des.schedule_at(
        failure_ns,
        kind="source_failure_after_commit",
        priority=EventPriority.FAILURE,
        callback=_record_gpu_failure(
            run, gpu=source_gpu, failure_id="source-after-commit"
        ),
    )
    run.des.run()
    extent = run.des.extents[extent_id]
    run.witnesses["source_after_commit"] = {
        "failure_ns": failure_ns,
        "hbm_transfer_state": hbm_transfer.state.value,
        "dram_transfer_state": dram_transfer.state.value,
        "hbm_copy_state": extent.locations[hbm_destination].value,
        "dram_copy_state": extent.locations[dram_destination].value,
        "source_state": extent.locations[source].value,
    }
    run.assert_contract(
        "source_failure_preserves_independent_committed_backups",
        not run.des.gpu_hbm[source_gpu].healthy
        and hbm_transfer.state is TransferState.COMMITTED
        and dram_transfer.state is TransferState.COMMITTED
        and extent.locations[hbm_destination] is KvCopyState.COMMITTED
        and extent.locations[dram_destination] is KvCopyState.COMMITTED
        and extent.locations[source] is KvCopyState.LOST,
        run.witnesses["source_after_commit"],
    )


_CONFIG_MANAGER_STATE_FIELDS = (
    "_raw",
    "_model",
    "_scheduler",
    "_platform",
    "_cache",
    "_hicache",
    "_predictor",
    "_communication",
    "_comm_predictor",
)


@contextmanager
def _p0_rotation_controller() -> Any:
    """Construct the real Router controller without leaking process globals.

    P0 does not start Scheduler processes.  The controller's real rotation
    table, budget publishers, topology checks, and stage command queues are
    nevertheless exercised.  ConfigManager and the in-process IPC registry
    are restored exactly so this contract cell cannot contaminate another
    test or simulator invocation.
    """

    config_state = {
        name: getattr(ConfigManager, name) for name in _CONFIG_MANAGER_STATE_FIELDS
    }
    queue_state = {
        endpoint: deque(values) for endpoint, values in _QUEUES.items()
    }
    controller: DataParallelController | None = None
    command_queues = [Queue() for _ in range(4)]
    try:
        ConfigManager._raw = {
            "model_path": "mock-model",
            "scheduler": {
                "dp_size": 4,
                "pp_size": 1,
                "tp_size": 1,
                "load_balance_method": "cache_aware",
            },
            "aegis_l1_l2": {
                "enabled": True,
                "committed_copy_protocol": True,
                "design_semantics": DESIGN_SEMANTICS,
                "host_alignment_tokens": 1,
                "topology": {
                    "dp_rank_to_node": [0, 1, 2, 3],
                    "pp_stages_per_node": 1,
                    "gpu_backup_requires_direct_retry_route": True,
                    "host_pool_scope": "node_shared_process_external",
                },
            },
        }
        for name in _CONFIG_MANAGER_STATE_FIELDS[1:]:
            setattr(ConfigManager, name, None)
        _QUEUES.clear()
        server_args = ServerArgs(dp_size=4, pp_size=1, tp_size=1)
        port_args = PortArgs.init_new(server_args)
        controller = DataParallelController(
            server_args,
            port_args,
            scheduler_cmd_queues=command_queues,
        )
        controller._aegis_committed_copy_protocol = True
        controller._aegis_design_semantics = DESIGN_SEMANTICS
        yield controller, command_queues
    finally:
        if controller is not None:
            controller.close_event_journal()
        _QUEUES.clear()
        _QUEUES.update(queue_state)
        for name, value in config_state.items():
            setattr(ConfigManager, name, value)


def _drain_queue(queue: Queue[Any]) -> list[Any]:
    values: list[Any] = []
    while True:
        try:
            values.append(queue.get_nowait())
        except Empty:
            return values


def _rotation_fabric(deployment: Mapping[str, Any]) -> CopyFabricConfig:
    resources = deployment["copy_resources"]
    links = resources["links"]
    pcie = links["pcie"]
    nvlink = links["nvlink_nvswitch"]
    nic = links["nic_rdma"]
    fabric = links["cross_node_fabric"]
    dram = resources["dram_lanes_per_node"]
    cross_node_setup = (
        2 * int(pcie["setup_ns"])
        + 2 * int(nic["setup_ns"])
        + int(fabric["setup_ns"])
    )
    host_setup = int(pcie["setup_ns"]) + int(dram["setup_ns"])
    return CopyFabricConfig(
        bandwidths=CopyFabricBandwidths(
            gpu_copy_bytes_per_s=int(nvlink["peak_bandwidth_bytes_per_s"]),
            nvlink_bytes_per_s=int(nvlink["peak_bandwidth_bytes_per_s"]),
            pcie_bytes_per_s=int(pcie["peak_bandwidth_bytes_per_s"]),
            nic_bytes_per_s=int(nic["peak_bandwidth_bytes_per_s"]),
            fabric_bytes_per_s=int(
                fabric["central_effective_bandwidth_bytes_per_s"]
            ),
            dram_write_bytes_per_s=int(dram["bandwidth_bytes_per_s"]),
            dram_read_bytes_per_s=int(dram["bandwidth_bytes_per_s"]),
        ),
        setup=CopyFabricSetupLatencies(
            same_node_d2d_ns=int(nvlink["setup_ns"]),
            cross_node_d2d_ns=cross_node_setup,
            same_node_d2h_ns=host_setup,
            cross_node_d2h_ns=cross_node_setup + int(dram["setup_ns"]),
            same_node_h2d_ns=host_setup,
            cross_node_h2d_ns=cross_node_setup + int(dram["setup_ns"]),
        ),
        fabric_id="p0-contract-fabric",
    )


def _install_rotation_hint(
    controller: DataParallelController,
    *,
    request_id: str,
    token_count: int,
    route_version: int,
) -> None:
    budget = controller._aegis_budget_publishers[1].current
    if budget is None:
        raise RuntimeError("P0 target DP has no published Aegis budget")
    stage = {
        "rid": request_id,
        "backup_key": request_id,
        "request_backup_key": request_id,
        "pp_rank": 0,
        "source_dp_rank": 0,
        "target_dp_rank": 1,
        "version": 1,
        "route_version": route_version,
        "budget_version": budget.budget_version,
        "token_count": token_count,
        "l1_tokens": token_count,
        "l2_tokens": token_count,
        "hbm_pending_tokens": 0,
        "dram_pending_tokens": 0,
        "reserved_l1_tokens": token_count,
        "reserved_l2_tokens": token_count,
        "safe_frontier_tokens": token_count,
        "recompute_tokens": 0,
        "materialization_scope": "committed_prefix",
        "source_committed_token_end": token_count,
        "group_commit_verified": True,
        "copy_completion_evidence": "p0_global_des_group_commit",
        "execution_id": f"{request_id}/execution-1",
        "block_generation": 0,
        "design_semantics": DESIGN_SEMANTICS,
        "failure_scope": "node",
        "host_pool_node_id": "node0",
        "host_pool_lease": {
            "node_id": 0,
            "backup_key": request_id,
            "version": 1,
            "tokens": token_count,
        },
        "gpu_candidate": {
            "valid": True,
            "target_dp_rank": 1,
            "reusable_dp_ranks": [1],
        },
        "host_candidate": {
            "valid": True,
            "host_pool_node_id": 0,
            "target_dp_rank": 1,
            "recovery_path": ["h2d"],
        },
        "tier_committed_source_frontiers": {
            "hbm": token_count,
            "dram": token_count,
        },
    }
    controller._aegis_stage_reservation_plans[(request_id, 1, 0)] = {
        **stage,
        "allowed_tiers": ["l1", "l2"],
        "required_pp_size": 1,
        "source_pp_rank": 0,
        "target_pp_rank": 0,
        "dp_budget_snapshot": budget.to_dict(),
    }
    controller._aegis_recovery_hints[request_id] = {
        **stage,
        "all_stages_committed": True,
        "required_stage_count": 1,
        "stage_payloads": [dict(stage)],
        "stage_hbm_frontiers": [token_count],
        "stage_dram_frontiers": [token_count],
    }


def _apply_replacement_hbm_ack(
    controller: DataParallelController,
    *,
    request_id: str,
    token_count: int,
    route_version: int,
    committed: bool,
) -> None:
    budget = controller._aegis_budget_publishers[2].current
    if budget is None:
        raise RuntimeError("P0 replacement DP has no published Aegis budget")
    payload = {
        "rid": request_id,
        "backup_key": request_id,
        "source_dp_rank": 0,
        "target_dp_rank": 2,
        "version": 1,
        "route_version": route_version,
        "budget_version": budget.budget_version,
        "ack": True,
        "l1_tokens": token_count if committed else 0,
        "hbm_pending_tokens": 0 if committed else token_count,
        "reserved_l1_tokens": token_count,
        "tier_committed_source_frontiers": {
            "hbm": token_count if committed else 0
        },
        "recompute_tokens": 0,
        "safe_frontier_tokens": token_count if committed else 0,
        "recovery_action": "l1_backup",
        "failure_scope": "node",
        "design_semantics": DESIGN_SEMANTICS,
        "gpu_candidate": {
            "valid": True,
            "target_dp_rank": 2,
            "reusable_dp_ranks": [2],
        },
    }
    event_type = (
        "aegis_l1_l2_group_commit_ack"
        if committed
        else "aegis_l1_l2_backup_ack"
    )
    if committed:
        payload.update(
            {
                "token_count": token_count,
                "materialization_scope": "committed_prefix",
                "source_committed_token_end": token_count,
                "group_commit_verified": True,
                "copy_completion_evidence": "p0_global_des_group_commit",
                "execution_id": f"{request_id}/execution-1",
                "block_generation": 0,
            }
        )
    controller.apply_aegis_rank_event(
        {
            "type": event_type,
            "payload": payload,
            "rank_state": {
                "normal_l1_pressure": 0.0,
                "normal_l2_pressure": 0.0,
            },
            "dp_rank": 2,
            "pp_rank": 0,
            "tp_rank": 0,
            "request_backup_key": request_id,
            "stage_backup_key": request_id,
            "source_pp_rank": 0,
            "target_pp_rank": 0,
            "required_pp_size": 1,
            "route_version": route_version,
            "budget_version": budget.budget_version,
        }
    )


def _scenario_target_first_and_node_failure(
    run: _ScenarioRun, deployment: Mapping[str, Any]
) -> None:
    gpus = tuple(
        _add_gpu(
            run,
            deployment,
            node=f"node{dp_rank}",
            local_gpu_id=0,
            dp_rank=dp_rank,
        )
        for dp_rank in range(4)
    )
    source_gpu, target_gpu, replacement_gpu, retry_gpu = gpus
    request_id = "p0-target-first-node-failure"
    version = 1
    route_version = 7
    token_count = int(deployment["aegiskv"]["prefix_chunk_tokens"])
    kv_bytes_per_token = int(
        deployment["models"]["qwen3-8b"]["kv_bytes_per_token_total"]
    )
    protected_bytes = token_count * kv_bytes_per_token
    _request_row(run, deployment, request_id=request_id, model="qwen3-8b")
    run.des.adjust_gpu_hbm(
        source_gpu,
        HbmAccount.PRIMARY_KV,
        protected_bytes,
        reason="p0_scheduler_primary_aggregate",
    )
    topology = OfflinePhysicalTopology(
        gpus=tuple(
            PhysicalGpuSpec(
                gpu=gpu,
                hbm_capacity_bytes=run.des.gpu_hbm[gpu].capacity_bytes,
                weights_bytes=run.des.gpu_hbm[gpu].weights,
                runtime_bytes=run.des.gpu_hbm[gpu].runtime,
            )
            for gpu in gpus
        ),
        nodes=tuple(
            NodeDramSpec(
                node_id=node,
                capacity_bytes=ledger.capacity_bytes,
                runtime_bytes=ledger.runtime,
            )
            for node, ledger in sorted(run.des.node_dram.items())
        ),
    )
    bridge = OfflineAegisTransferBridge(
        run.des,
        topology,
        fabric=_rotation_fabric(deployment),
    )
    source_event = {
        "type": "aegis_source_chunk_committed",
        "rid": request_id,
        "request_backup_key": request_id,
        "source_dp_rank": 0,
        "target_dp_rank": 1,
        "source_pp_rank": 0,
        "target_pp_rank": 0,
        "required_pp_size": 1,
        "version": version,
        "route_version": route_version,
        "budget_version": 1,
        "source_committed_token_end": token_count,
        "reserved_token_count": token_count,
        "prefix_chunk_tokens": token_count,
        "decode_chunk_tokens": int(
            deployment["aegiskv"]["sealed_decode_chunk_tokens"]
        ),
        "execution_id": f"{request_id}/execution-1",
        "block_generation": 0,
    }
    state: dict[str, Any] = {}

    with _p0_rotation_controller() as (controller, command_queues):
        _install_rotation_hint(
            controller,
            request_id=request_id,
            token_count=token_count,
            route_version=route_version,
        )
        initial_hint = controller.recovery_hint(request_id)
        if initial_hint is None:
            raise RuntimeError("P0 rotation fixture did not publish its hint")
        initial_plan = bridge.consume_source_chunk_committed(
            source_event,
            initial_hint,
            kv_bytes_per_token_by_tp_rank={0: kv_bytes_per_token},
            model_revision="Qwen/Qwen3-8B@p0-contract",
            kv_abi="sglsim-p0-bf16-kv-v1",
            parallel_layout="tp1-pp1",
        )
        run.des.run()
        initial_hbm = next(
            group
            for group in initial_plan.tier_groups
            if group.tier is StorageTier.GPU_AEGIS
        )
        initial_dram = next(
            group
            for group in initial_plan.tier_groups
            if group.tier is StorageTier.NODE_DRAM
        )
        dram_before_rotation = run.des.node_dram["node0"].aegis_committed
        if dram_before_rotation != protected_bytes:
            raise RuntimeError("P0 initial source-owned DRAM did not commit")

        def target_failure(des: GlobalDES, _event: Any) -> None:
            affected = des.fail_gpu(target_gpu)
            controller.mark_rank_hard_failed(1)
            invalidation = controller.invalidate_aegis_physical_failure(
                failed_dp_ranks=(1,),
                reason="p0_target_before_source",
            )
            rotations = invalidation["target_rotations"]
            if len(rotations) != 1:
                raise RuntimeError("P0 target failure did not rotate exactly once")
            rotation = dict(rotations[0])
            fenced_hint = controller.recovery_hint(request_id)
            if fenced_hint is None:
                raise RuntimeError("P0 rotation fence lost the recovery hint")
            fence = bridge.consume_source_chunk_committed(
                source_event,
                fenced_hint,
                kv_bytes_per_token_by_tp_rank={0: kv_bytes_per_token},
                model_revision="Qwen/Qwen3-8B@p0-contract",
                kv_abi="sglsim-p0-bf16-kv-v1",
                parallel_layout="tp1-pp1",
            )
            rotated_commands = _drain_queue(command_queues[2])
            _apply_replacement_hbm_ack(
                controller,
                request_id=request_id,
                token_count=token_count,
                route_version=route_version,
                committed=False,
            )
            activated_hint = controller.recovery_hint(request_id)
            if activated_hint is None:
                raise RuntimeError("P0 replacement ACK lost the recovery hint")
            replacement = bridge.consume_source_chunk_committed(
                source_event,
                activated_hint,
                kv_bytes_per_token_by_tp_rank={0: kv_bytes_per_token},
                model_revision="Qwen/Qwen3-8B@p0-contract",
                kv_abi="sglsim-p0-bf16-kv-v1",
                parallel_layout="tp1-pp1",
            )
            state.update(
                {
                    "rotation": rotation,
                    "fence": fence,
                    "replacement_plan": replacement,
                    "rotated_commands": rotated_commands,
                    "target_affected_extents": list(affected),
                }
            )
            run.failures.append(
                {
                    "schema": "aegiskv.p0.failure.v1",
                    "failure_id": "target-gpu-first",
                    "failure_ns": des.now_ns,
                    "domain": "gpu",
                    "target": target_gpu.key,
                    "affected_extents": list(affected),
                }
            )
            run.rank_steps.append(
                {
                    "schema": "aegiskv.p0.physical-rank-step.v1",
                    "step_id": "target-rotation/fenced-and-readmitted",
                    "timestamp_ns": des.now_ns,
                    "phase": "standby_target_rotated",
                    "dp_rank": 2,
                    "pp_rank": 0,
                    "tp_rank": 0,
                    "route_version": route_version,
                    "previous_target_dp_rank": 1,
                    "replacement_target_dp_rank": 2,
                    "hbm_only_readmission": True,
                }
            )

        target_failure_ns = run.des.now_ns + 1
        run.des.schedule_at(
            target_failure_ns,
            kind="target_gpu_failure_and_route_rotation",
            priority=EventPriority.FAILURE,
            callback=target_failure,
        )
        run.des.run()
        _apply_replacement_hbm_ack(
            controller,
            request_id=request_id,
            token_count=token_count,
            route_version=route_version,
            committed=True,
        )
        replacement_hint = controller.recovery_hint(request_id)
        if replacement_hint is None:
            raise RuntimeError("P0 replacement commit lost the recovery hint")
        replacement_group = next(
            group
            for group in state["replacement_plan"].tier_groups
            if group.tier is StorageTier.GPU_AEGIS
        )
        dram_after_rotation = run.des.node_dram["node0"].aegis_committed
        owners_after_rotation = {
            row["group_id"]: row for row in bridge.export()["group_owners"]
        }
        jobs_by_group = {
            group_id: [
                job for job in bridge.jobs if job.group_id == group_id
            ]
            for group_id in (initial_hbm.group_id, replacement_group.group_id)
        }
        run.rank_steps.append(
            {
                "schema": "aegiskv.p0.physical-rank-step.v1",
                "step_id": "target-rotation/replacement-committed",
                "timestamp_ns": run.des.now_ns,
                "phase": "replacement_hbm_committed",
                "dp_rank": 2,
                "pp_rank": 0,
                "tp_rank": 0,
                "route_version": route_version,
                "group_id": replacement_group.group_id,
                "committed_bytes": run.des.gpu_hbm[
                    replacement_gpu
                ].aegis_committed,
            }
        )

        def source_node_failure(des: GlobalDES, _event: Any) -> None:
            affected = des.fail_node("node0")
            controller.mark_rank_hard_failed(0)
            invalidation = controller.invalidate_aegis_physical_failure(
                failed_dp_ranks=(0,),
                failed_node_ids=(0,),
                reason="p0_source_node_failure",
            )
            state["source_invalidation"] = invalidation
            run.failures.append(
                {
                    "schema": "aegiskv.p0.failure.v1",
                    "failure_id": "source-node-second",
                    "failure_ns": des.now_ns,
                    "domain": "node",
                    "target": "node0",
                    "affected_extents": list(affected),
                }
            )

        source_failure_ns = run.des.now_ns + 1
        run.des.schedule_at(
            source_failure_ns,
            kind="source_node_failure_after_rotation",
            priority=EventPriority.FAILURE,
            callback=source_node_failure,
        )
        run.des.run()
        source_failed_hint = controller.recovery_hint(request_id)
        if source_failed_hint is None:
            raise RuntimeError("P0 source failure lost the recovery hint")
        recovery_plan = bridge.plan_failover_recovery(
            retry_request_id=f"{request_id}/retry-hbm",
            request_backup_key=request_id,
            retry_dp_rank=3,
            tier_tokens={
                StorageTier.GPU_AEGIS: int(source_failed_hint["l1_tokens"]),
                StorageTier.NODE_DRAM: int(source_failed_hint["l2_tokens"]),
            },
            route_version=route_version,
        )
        run.des.run()
        recovery_outcomes = {
            job.recovery_id: bridge.export()["recovery_outcomes"][job.recovery_id]
            for job in bridge.recovery_jobs
            if job.retry_request_id == f"{request_id}/retry-hbm"
        }
        run.rank_steps.append(
            {
                "schema": "aegiskv.p0.physical-rank-step.v1",
                "step_id": "source-node-failure/healthy-hbm-recovery",
                "timestamp_ns": run.des.now_ns,
                "phase": "healthy_compatible_tier_recovery",
                "dp_rank": 3,
                "pp_rank": 0,
                "tp_rank": 0,
                "route_version": route_version,
                "selected_tier": recovery_plan["selected_tier"],
                "recovery_job_count": len(recovery_plan["jobs"]),
            }
        )

        def replacement_node_failure(des: GlobalDES, _event: Any) -> None:
            affected = des.fail_node("node2")
            controller.mark_rank_hard_failed(2)
            invalidation = controller.invalidate_aegis_physical_failure(
                failed_dp_ranks=(2,),
                failed_node_ids=(2,),
                reason="p0_replacement_node_failure",
            )
            state["replacement_invalidation"] = invalidation
            run.failures.append(
                {
                    "schema": "aegiskv.p0.failure.v1",
                    "failure_id": "replacement-node-third",
                    "failure_ns": des.now_ns,
                    "domain": "node",
                    "target": "node2",
                    "affected_extents": list(affected),
                }
            )

        replacement_failure_ns = run.des.now_ns + 1
        run.des.schedule_at(
            replacement_failure_ns,
            kind="replacement_node_failure_fail_closed",
            priority=EventPriority.FAILURE,
            callback=replacement_node_failure,
        )
        run.des.run()
        stale_recovery_error = ""
        try:
            bridge.plan_failover_recovery(
                retry_request_id=f"{request_id}/retry-stale-hbm",
                request_backup_key=request_id,
                retry_dp_rank=3,
                tier_tokens={StorageTier.GPU_AEGIS: token_count},
                route_version=route_version,
            )
        except ValueError as exc:
            stale_recovery_error = str(exc)
        terminal_hint = controller.recovery_hint(request_id)
        if terminal_hint is None:
            raise RuntimeError("P0 terminal failure lost the recovery hint")
        recompute_plan = bridge.plan_failover_recovery(
            retry_request_id=f"{request_id}/retry-recompute",
            request_backup_key=request_id,
            retry_dp_rank=3,
            tier_tokens={},
            route_version=route_version,
        )
        run.rank_steps.append(
            {
                "schema": "aegiskv.p0.physical-rank-step.v1",
                "step_id": "replacement-node-failure/stale-tier-rejected",
                "timestamp_ns": run.des.now_ns,
                "phase": "unhealthy_tier_fail_closed",
                "dp_rank": 3,
                "pp_rank": 0,
                "tp_rank": 0,
                "route_version": route_version,
                "selected_tier": recompute_plan["selected_tier"],
                "stale_recovery_rejected": bool(stale_recovery_error),
            }
        )

    rotation = state["rotation"]
    old_owner = owners_after_rotation[initial_hbm.group_id]
    new_owner = owners_after_rotation[replacement_group.group_id]
    old_job = jobs_by_group[initial_hbm.group_id][0]
    new_job = jobs_by_group[replacement_group.group_id][0]
    old_group = run.des.commit_groups[initial_hbm.group_id]
    new_group = run.des.commit_groups[replacement_group.group_id]
    dram_group = run.des.commit_groups[initial_dram.group_id]
    bridge_errors = bridge.audit_invariants()
    witness = {
        "failure_order": [row["failure_id"] for row in run.failures],
        "failure_ns": [row["failure_ns"] for row in run.failures],
        "version": version,
        "route_version": route_version,
        "rotation": rotation,
        "fence_duplicate": state["fence"].duplicate,
        "rotated_plan_commands": [
            {
                "operation": operation,
                "target_dp_rank": payload["target_dp_rank"],
                "allowed_tiers": list(payload["allowed_tiers"]),
                "route_version": payload["route_version"],
                "version": payload["version"],
                "target_rotation_hbm_only": payload[
                    "target_rotation_hbm_only"
                ],
            }
            for operation, payload in state["rotated_commands"]
        ],
        "old_hbm_owner": old_owner,
        "replacement_hbm_owner": new_owner,
        "old_hbm_group_state": old_group.state.value,
        "replacement_hbm_group_state": new_group.state.value,
        "old_hbm_transfer_state": run.des.transfers[
            old_job.transfer_id
        ].state.value,
        "replacement_hbm_transfer_state": run.des.transfers[
            new_job.transfer_id
        ].state.value,
        "old_hbm_destination_dp_rank": old_job.destination.gpu.dp_rank,
        "replacement_hbm_destination_dp_rank": new_job.destination.gpu.dp_rank,
        "dram_group_id": initial_dram.group_id,
        "dram_group_state": dram_group.state.value,
        "dram_job_count": sum(
            job.tier is StorageTier.NODE_DRAM for job in bridge.jobs
        ),
        "dram_committed_bytes_before_rotation": dram_before_rotation,
        "dram_committed_bytes_after_rotation": dram_after_rotation,
        "source_failed_hint": {
            "route_version": source_failed_hint["route_version"],
            "l1_tokens": source_failed_hint["l1_tokens"],
            "l2_tokens": source_failed_hint["l2_tokens"],
            "safe_frontier_tokens": source_failed_hint["safe_frontier_tokens"],
            "recovery_action": source_failed_hint["recovery_action"],
        },
        "healthy_recovery": {
            "selected_tier": recovery_plan["selected_tier"],
            "requested_tokens": recovery_plan["requested_tokens"],
            "job_sources": [job["source"] for job in recovery_plan["jobs"]],
            "job_destinations": [
                job["destination"] for job in recovery_plan["jobs"]
            ],
            "outcomes": recovery_outcomes,
        },
        "terminal_hint": {
            "route_version": terminal_hint["route_version"],
            "l1_tokens": terminal_hint["l1_tokens"],
            "l2_tokens": terminal_hint["l2_tokens"],
            "safe_frontier_tokens": terminal_hint["safe_frontier_tokens"],
            "recovery_action": terminal_hint["recovery_action"],
        },
        "stale_recovery_error": stale_recovery_error,
        "fail_closed_plan": recompute_plan,
        "replacement_rotations_after_source_loss": state[
            "replacement_invalidation"
        ]["target_rotations"],
        "bridge_audit_errors": bridge_errors,
    }
    run.witnesses["target_first_node_failure"] = witness
    run.assert_contract(
        "target_failure_rotates_same_route_without_owner_relabelling",
        rotation["source_dp_rank"] == 0
        and rotation["previous_target_dp_rank"] == 1
        and rotation["replacement_target_dp_rank"] == 2
        and rotation["version"] == version
        and rotation["route_version"] == route_version
        and rotation["route_version_changed"] is False
        and old_owner["target_dp_rank"] == 1
        and old_owner["released"] is True
        and new_owner["target_dp_rank"] == 2
        and old_owner["group_id"] != new_owner["group_id"]
        and old_job.destination.gpu == target_gpu
        and new_job.destination.gpu == replacement_gpu,
        witness,
    )
    run.assert_contract(
        "replacement_is_hbm_only_and_source_owned_dram_is_unchanged",
        state["fence"].duplicate
        and len(state["rotated_commands"]) == 1
        and state["rotated_commands"][0][0] == "aegis_l1_l2_backup_plan"
        and state["rotated_commands"][0][1]["allowed_tiers"] == ["l1"]
        and state["rotated_commands"][0][1][
            "target_rotation_hbm_only"
        ]
        is True
        and {group.tier for group in state["replacement_plan"].tier_groups}
        == {StorageTier.GPU_AEGIS}
        and new_group.state is CommitGroupState.COMMITTED
        and dram_group.state is CommitGroupState.COMMITTED
        and dram_before_rotation == dram_after_rotation == protected_bytes
        and witness["dram_job_count"] == 1,
        witness,
    )
    run.assert_contract(
        "source_node_failure_recovers_only_from_healthy_replacement_hbm",
        source_failure_ns > target_failure_ns
        and recovery_plan["selected_tier"] == StorageTier.GPU_AEGIS.value
        and recovery_plan["requested_tokens"] == token_count
        and source_failed_hint["l1_tokens"] == token_count
        and source_failed_hint["l2_tokens"] == 0
        and bool(recovery_outcomes)
        and set(recovery_outcomes.values()) == {"COMPLETED"}
        and all(
            job["source"]["tier"] == StorageTier.GPU_AEGIS.value
            and job["source"]["gpu"]["dp_rank"] == 2
            and job["destination"]["gpu"]["dp_rank"] == retry_gpu.dp_rank
            for job in recovery_plan["jobs"]
        ),
        witness,
    )
    run.assert_contract(
        "unhealthy_or_absent_tiers_fail_closed_to_recompute",
        replacement_failure_ns > source_failure_ns
        and terminal_hint["l1_tokens"] == 0
        and terminal_hint["l2_tokens"] == 0
        and terminal_hint["safe_frontier_tokens"] == 0
        and bool(stale_recovery_error)
        and recompute_plan["selected_tier"] == "recompute"
        and recompute_plan["jobs"] == []
        and state["replacement_invalidation"]["target_rotations"] == []
        and bridge_errors == [],
        witness,
    )


def _scenario_stable_same_timestamp(
    run: _ScenarioRun, deployment: Mapping[str, Any]
) -> None:
    gpu = _add_gpu(
        run, deployment, node="node0", local_gpu_id=0, dp_rank=0
    )
    _ensure_node(run, deployment, "node1")
    _request_row(
        run,
        deployment,
        request_id="p0-stable-same-timestamp",
        model="qwen3-8b",
    )
    observed: list[str] = []
    timestamp = int(deployment["aegiskv"]["sealed_decode_chunk_tokens"])

    def record(label: str) -> Callable[[GlobalDES, Any], None]:
        def callback(des: GlobalDES, _event: Any) -> None:
            observed.append(label)
            run.rank_steps.append(
                {
                    "schema": "aegiskv.p0.physical-rank-step.v1",
                    "step_id": f"tie/{label}",
                    "timestamp_ns": des.now_ns,
                    "phase": "same_timestamp_dispatch",
                    "dp_rank": 0,
                    "pp_rank": 0,
                    "tp_rank": 0,
                    "label": label,
                }
            )

        return callback

    run.des.schedule_at(
        timestamp,
        kind="same_timestamp_control_first",
        priority=EventPriority.CONTROL,
        callback=record("control-first"),
    )

    def failure(des: GlobalDES, event: Any) -> None:
        observed.append("failure")
        _record_gpu_failure(
            run, gpu=gpu, failure_id="same-timestamp-failure"
        )(des, event)

    run.des.schedule_at(
        timestamp,
        kind="same_timestamp_failure",
        priority=EventPriority.FAILURE,
        callback=failure,
    )
    run.des.schedule_at(
        timestamp,
        kind="same_timestamp_control_second",
        priority=EventPriority.CONTROL,
        callback=record("control-second"),
    )
    run.des.run()
    run.witnesses["same_timestamp_order"] = observed
    run.assert_contract(
        "same_timestamp_priority_then_sequence",
        observed == ["failure", "control-first", "control-second"],
        {"timestamp_ns": timestamp, "observed_order": observed},
    )


_SCENARIO_HANDLERS: dict[
    str, Callable[[_ScenarioRun, Mapping[str, Any]], None]
] = {
    "independent_dp_overlap": _scenario_independent_dp_overlap,
    "qwen3_8b_tp1_shard": lambda run, deployment: _scenario_qwen_tp(
        run, deployment, model_name="qwen3-8b"
    ),
    "qwen3_32b_tp4_shard": lambda run, deployment: _scenario_qwen_tp(
        run, deployment, model_name="qwen3-32b"
    ),
    "synthetic_pp2_shard": _scenario_synthetic_pp2,
    "same_node_peer_hbm_path": lambda run, deployment: _scenario_path(
        run, deployment, path="same_node_peer_hbm"
    ),
    "cross_node_peer_hbm_path": lambda run, deployment: _scenario_path(
        run, deployment, path="cross_node_peer_hbm"
    ),
    "source_to_node_dram_path": lambda run, deployment: _scenario_path(
        run, deployment, path="source_to_node_dram"
    ),
    "node_dram_to_recovery_gpu_path": lambda run, deployment: _scenario_path(
        run, deployment, path="node_dram_to_recovery_gpu"
    ),
    "failure_in_pending_window": _scenario_failure_in_pending,
    "failure_after_commit_source_first": _scenario_failure_after_commit,
    "target_first_and_node_failure": _scenario_target_first_and_node_failure,
    "stable_same_timestamp_tie_hash": _scenario_stable_same_timestamp,
}


def _execute_scenario(
    scenario: str,
    *,
    seed: int,
    deployment: Mapping[str, Any],
) -> _ScenarioRun:
    if scenario not in _SCENARIO_HANDLERS:
        raise RuntimeError(f"unsupported frozen P0 scenario {scenario!r}")
    run = _ScenarioRun(scenario=scenario, seed=int(seed))
    _SCENARIO_HANDLERS[scenario](run, deployment)
    if run.des.has_events():
        run.des.run()
    return run


def _transition_check(des: GlobalDES) -> tuple[bool, dict[str, Any]]:
    allowed = {
        ("QUEUED", "IN_FLIGHT"),
        ("QUEUED", "ABORTED"),
        ("IN_FLIGHT", "ACKED"),
        ("IN_FLIGHT", "COMMITTED"),
        ("IN_FLIGHT", "ABORTED"),
        ("ACKED", "COMMITTED"),
        ("ACKED", "ABORTED"),
    }
    histories: dict[str, list[str]] = {}
    for row in des.export_timeseries()["transfers"]:
        state = str(row["state"])
        history = histories.setdefault(str(row["transfer_id"]), [])
        if not history or history[-1] != state:
            history.append(state)
    illegal: list[dict[str, str]] = []
    for transfer_id, history in sorted(histories.items()):
        for before, after in zip(history, history[1:]):
            if (before, after) not in allowed:
                illegal.append(
                    {"transfer_id": transfer_id, "before": before, "after": after}
                )
    return not illegal, {"histories": histories, "illegal_edges": illegal}


def _required_invariant_records(
    run: _ScenarioRun,
    *,
    deterministic_hash: str,
    replay_hash: str,
    deployment: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    des = run.des
    exported = des.export_timeseries()
    strict_errors = des.audit_invariants(strict_extent_accounting=True)
    journal_times = [int(row["timestamp_ns"]) for row in exported["journal"]]
    gpu_peaks: dict[str, int] = {}
    for row in exported["gpu_hbm"]:
        key = (
            f"{row['node_id']}/gpu{row['local_gpu_id']}/dp{row['dp_rank']}/"
            f"pp{row['pp_rank']}/tp{row['tp_rank']}"
        )
        gpu_peaks[key] = max(gpu_peaks.get(key, 0), int(row["used_bytes"]))
    dram_peaks: dict[str, int] = {}
    for row in exported["node_dram"]:
        node = str(row["node_id"])
        dram_peaks[node] = max(dram_peaks.get(node, 0), int(row["used_bytes"]))
    gpu_capacity_ok = all(
        ledger.used_bytes <= ledger.capacity_bytes for ledger in des.gpu_hbm.values()
    )
    dram_capacity_ok = all(
        ledger.used_bytes <= ledger.capacity_bytes for ledger in des.node_dram.values()
    )
    transition_ok, transition_evidence = _transition_check(des)
    group_failures = [
        group.group_id
        for group in des.commit_groups.values()
        if group.state is CommitGroupState.COMMITTED
        and (
            set(group.acked_shards) != set(group.expected_shard_keys)
            or set(group.shard_transfers) != set(group.expected_shard_keys)
        )
    ]
    forbidden_edges = [
        transfer.transfer_id
        for transfer in des.transfers.values()
        if transfer.source.tier is StorageTier.GPU_AEGIS
        and transfer.destination.tier is StorageTier.NODE_DRAM
    ]
    admission = deployment["aegiskv"]["tier_admission"]
    tier_groups = {
        "gpu_aegis": sum(
            group.tier is StorageTier.GPU_AEGIS
            for group in des.commit_groups.values()
        ),
        "node_dram": sum(
            group.tier is StorageTier.NODE_DRAM
            for group in des.commit_groups.values()
        ),
    }
    overlap = run.witnesses.get("non_lockstep_dp_overlap")
    non_lockstep_supported = (
        deployment["global_des"]["kernel"] == "single_authoritative_min_heap"
        and deployment["global_des"]["lockstep_rank_clock_forbidden"] is True
    )
    if overlap is not None:
        non_lockstep_supported = non_lockstep_supported and (
            overlap["max_ns"] != overlap["sum_ns"]
            and max(overlap["completion_ns"].values()) == overlap["max_ns"]
        )

    records: dict[str, dict[str, Any]] = {
        "global_event_order": {
            "passed": journal_times == sorted(journal_times),
            "evidence": {
                "journal_rows": len(journal_times),
                "nondecreasing_timestamp_ns": journal_times == sorted(journal_times),
            },
        },
        "deterministic_event_hash": {
            "passed": deterministic_hash == replay_hash,
            "evidence": {
                "event_content_sha256": deterministic_hash,
                "replay_event_content_sha256": replay_hash,
                "excluded_from_hash": ["run_directory", "run_id", "lineage"],
            },
        },
        "per_gpu_hbm_capacity": {
            "passed": gpu_capacity_ok and bool(des.gpu_hbm),
            "evidence": {
                "physical_gpu_count": len(des.gpu_hbm),
                "peak_used_bytes": dict(sorted(gpu_peaks.items())),
            },
        },
        "per_node_dram_capacity": {
            "passed": dram_capacity_ok and bool(des.node_dram),
            "evidence": {
                "physical_node_count": len(des.node_dram),
                "peak_used_bytes": dict(sorted(dram_peaks.items())),
            },
        },
        "kv_extent_conservation": {
            "passed": not strict_errors,
            "evidence": {
                "strict_extent_accounting": True,
                "audit_errors": strict_errors,
                "extent_count": len(des.extents),
            },
        },
        "legal_state_transition": {
            "passed": transition_ok and not strict_errors,
            "evidence": transition_evidence,
        },
        "all_shards_ack_before_commit": {
            "passed": not group_failures,
            "evidence": {
                "commit_group_count": len(des.commit_groups),
                "invalid_committed_groups": group_failures,
                "groups": {
                    group.group_id: {
                        "state": group.state.value,
                        "expected": list(group.expected_shard_keys),
                        "acked": sorted(group.acked_shards),
                    }
                    for group in sorted(
                        des.commit_groups.values(), key=lambda item: item.group_id
                    )
                },
            },
        },
        "no_hbm_to_dram_edge": {
            "passed": not forbidden_edges,
            "evidence": {
                "forbidden_transfer_ids": forbidden_edges,
                "configured_forbidden_edges": list(
                    deployment["aegiskv"]["forbidden_copy_edges"]
                ),
            },
        },
        "independent_hbm_dram_admission": {
            "passed": admission["peer_hbm"] == "independent"
            and admission["node_dram"] == "independent"
            and admission["same_source_production_buffer"] is True,
            "evidence": {
                "configured_admission": copy.deepcopy(admission),
                "observed_tier_commit_groups": tier_groups,
                "coupled_group_type_present": False,
            },
        },
        "non_lockstep_dp_overlap": {
            "passed": non_lockstep_supported,
            "evidence": {
                "kernel": deployment["global_des"]["kernel"],
                "lockstep_rank_clock_forbidden": deployment["global_des"][
                    "lockstep_rank_clock_forbidden"
                ],
                "scenario_overlap_witness": overlap,
            },
        },
    }
    if tuple(records) != REQUIRED_INVARIANTS:
        raise AssertionError("runner invariant table drifted from frozen P0 gate")
    return records


def run_p0(
    run_dir: str | os.PathLike[str],
    *,
    config_root: str | os.PathLike[str] = CONFIG_ROOT,
) -> dict[str, Any]:
    """Run all twelve frozen P0 cells into a caller-supplied fresh directory."""

    target = Path(run_dir)
    if os.path.lexists(os.fspath(target)):
        raise FileExistsError(f"P0 run directory already exists: {target}")

    suite = load_suite(Path(config_root))
    p0 = suite.group_map[GROUP]
    cells = tuple(p0.iter_cells())
    scenarios = tuple(str(cell["axes"]["contract_scenario"]) for cell in cells)
    if len(cells) != 12 or scenarios != P0_SCENARIOS:
        raise RuntimeError("P0 must mechanically resolve to the frozen 12 scenarios")
    if any(int(cell["axes"]["seed"]) != 2026082900 for cell in cells):
        raise RuntimeError("P0 seed drifted from the frozen JSON axis")
    if set(_SCENARIO_HANDLERS) != set(P0_SCENARIOS):
        missing = sorted(set(P0_SCENARIOS) - set(_SCENARIO_HANDLERS))
        extra = sorted(set(_SCENARIO_HANDLERS) - set(P0_SCENARIOS))
        raise RuntimeError(
            f"P0 scenario implementation mismatch: missing={missing}, extra={extra}"
        )

    target.mkdir(parents=True, exist_ok=False)
    p0_root = target / GROUP
    p0_root.mkdir(exist_ok=False)
    code_hash = _implementation_hash(Path(config_root).resolve())
    cell_summaries: list[dict[str, Any]] = []

    for cell in cells:
        scenario = str(cell["axes"]["contract_scenario"])
        seed = int(cell["axes"]["seed"])
        resolved = _resolved_three_layers(p0, scenario=scenario, seed=seed)
        deployment = resolved["deployment"]
        primary = _execute_scenario(
            scenario, seed=seed, deployment=deployment
        )
        replay = _execute_scenario(
            scenario, seed=seed, deployment=deployment
        )
        event_hash = _event_content_hash(primary.des)
        replay_hash = _event_content_hash(replay.des)
        invariants = _required_invariant_records(
            primary,
            deterministic_hash=event_hash,
            replay_hash=replay_hash,
            deployment=deployment,
        )
        assertions_pass = bool(primary.assertions) and all(
            record["passed"] for record in primary.assertions.values()
        )
        invariants_pass = set(invariants) == set(REQUIRED_INVARIANTS) and all(
            record["passed"] for record in invariants.values()
        )
        cell_passed = assertions_pass and invariants_pass
        audit_details = {
            "schema": AUDIT_DETAILS_SCHEMA,
            "group": GROUP,
            "cell_ordinal": int(cell["cell_ordinal"]),
            "contract_scenario": scenario,
            "seed": seed,
            "status": "PASS" if cell_passed else "FAIL",
            "required_invariants": invariants,
            "scenario_assertions": copy.deepcopy(primary.assertions),
            "event_content_sha256": event_hash,
            "claims": {
                "contract_execution_only": True,
                "full_scheduler_integration": False,
                "controller_bridge_target_rotation_closed_loop": (
                    scenario == "target_first_and_node_failure"
                ),
                "performance_result": False,
                "real_hardware_measurement": False,
            },
        }
        cell_dir = p0_root / scenario
        write_global_des_artifacts(
            cell_dir,
            des=primary.des,
            rank_steps=primary.rank_steps,
            requests=primary.requests,
            failures=primary.failures,
            resolved_config=resolved,
            lineage_metadata={
                "engine": ENGINE,
                "code_hash": code_hash,
                "suite_id": suite.suite["suite_id"],
                "group": GROUP,
                "contract_scenario": scenario,
                "execution_scope": "executed_contract_cell_only",
            },
            audit_details=audit_details,
        )
        manifest = json.loads((cell_dir / "manifest.json").read_text())
        cell_summaries.append(
            {
                "cell_ordinal": int(cell["cell_ordinal"]),
                "contract_scenario": scenario,
                "status": "PASS" if cell_passed else "FAIL",
                "event_content_sha256": event_hash,
                "manifest_payload_sha256": manifest[
                    "canonical_payload_sha256"
                ],
                "required_invariants_passed": sum(
                    record["passed"] for record in invariants.values()
                ),
                "required_invariants_total": len(REQUIRED_INVARIANTS),
                "scenario_assertions_passed": sum(
                    record["passed"] for record in primary.assertions.values()
                ),
                "scenario_assertions_total": len(primary.assertions),
            }
        )

    all_cells_pass = (
        len(cell_summaries) == 12
        and tuple(row["contract_scenario"] for row in cell_summaries)
        == P0_SCENARIOS
        and all(row["status"] == "PASS" for row in cell_summaries)
        and all(
            row["required_invariants_passed"]
            == row["required_invariants_total"]
            == len(REQUIRED_INVARIANTS)
            for row in cell_summaries
        )
    )
    rotation_cell = next(
        row
        for row in cell_summaries
        if row["contract_scenario"] == "target_first_and_node_failure"
    )
    rotation_gate_passed = (
        rotation_cell["status"] == "PASS"
        and rotation_cell["scenario_assertions_passed"]
        == rotation_cell["scenario_assertions_total"]
        == 4
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "suite_id": suite.suite["suite_id"],
        "configured_suite_status": PLANNED_STATUS,
        "group": GROUP,
        "gate_mode": suite.suite["p0_gate"]["mode"],
        "status": "PASS" if all_cells_pass else "FAIL",
        "executed_contract_cells": len(cell_summaries),
        "required_contract_cells": 12,
        "required_invariants": list(REQUIRED_INVARIANTS),
        "evidence_class": EVIDENCE_CLASS,
        "engine": ENGINE,
        "code_hash": code_hash,
        "resolved_p0_config_sha256": config_hash(
            {layer.layer: layer.resolved for layer in p0.layers}
        ),
        "cells": cell_summaries,
        "claims": {
            "contract_execution_only": True,
            "full_scheduler_integration": False,
            "controller_bridge_target_rotation_closed_loop": (
                rotation_gate_passed
            ),
            "performance_result": False,
            "real_hardware_measurement": False,
            # Passing this executable-kernel contract is necessary but not
            # sufficient for paper experiments.  P1--P12 additionally require
            # the real OfflineSchedulerCluster/Aegis transfer path to pass its
            # integration gate; never promote a standalone GlobalDES contract
            # into an execution-readiness claim.
            "configured_p0_contract_gate_passed": all_cells_pass,
            "p1_p12_execution_ready": False,
        },
    }
    with (p0_root / "summary.json").open("xb") as handle:
        handle.write(_canonical_bytes(summary))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen 12-cell AegisKV global-DES P0 contract gate"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="fresh output directory; cells are written below RUN_DIR/P0",
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=CONFIG_ROOT,
        help="v3 configuration root (defaults to the repository-frozen suite)",
    )
    args = parser.parse_args(argv)
    try:
        summary = run_p0(args.run_dir, config_root=args.config_root)
    except Exception as exc:
        print(f"P0 contract runner failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": summary["status"],
                "executed_contract_cells": summary["executed_contract_cells"],
                "summary": str(args.run_dir / GROUP / "summary.json"),
            },
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "PASS" else 1


__all__ = ["AUDIT_DETAILS_SCHEMA", "ENGINE", "SUMMARY_SCHEMA", "main", "run_p0"]
