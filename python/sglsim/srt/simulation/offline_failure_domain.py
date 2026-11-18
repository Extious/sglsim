"""Resolve offline fail-stop declarations to exact physical identities.

The offline Scheduler currently executes one metadata process for every
``(dp_rank, pp_rank)`` and mirrors a symmetric TP-rank view onto every
physical TP shard.  A failure declaration therefore must not be interpreted
as merely a DP-rank integer: the same integer means different physical loss
for a worker, process, GPU, and node failure.

This module is deliberately side-effect free.  It is the target-resolution
boundary used before a failure is placed on the global event heap.  Keeping
resolution separate from injection makes it possible to reject a topology
that the current process adapter cannot faithfully mutate, rather than
silently widening a single-GPU/process failure into a whole-worker failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from sglsim.srt.simulation.global_des import PhysicalGpuId
from sglsim.srt.simulation.offline_resource_bridge import OfflinePhysicalTopology


OFFLINE_FAILURE_TARGET_SCHEMA = "sglsim.offline.failure-target.v1"


class OfflineFailureScope(str, Enum):
    """Physical fail-stop domains exposed by the experiment schema."""

    WORKER = "worker"
    PROCESS = "process"
    GPU = "gpu"
    NODE = "node"


@dataclass(frozen=True, order=True)
class OfflineSchedulerProcessId:
    """One offline Scheduler process representing a complete TP group."""

    dp_rank: int
    pp_rank: int

    def __post_init__(self) -> None:
        if int(self.dp_rank) < 0 or int(self.pp_rank) < 0:
            raise ValueError("Scheduler process ranks must be nonnegative")

    @property
    def key(self) -> str:
        return f"dp{self.dp_rank}/pp{self.pp_rank}"

    def as_dict(self) -> dict[str, int]:
        return {"dp_rank": self.dp_rank, "pp_rank": self.pp_rank}


@dataclass(frozen=True)
class ResolvedOfflineFailureTarget:
    """Exact resource and service impact of one fail-stop declaration."""

    scope: OfflineFailureScope
    physical_gpus: tuple[PhysicalGpuId, ...]
    scheduler_processes: tuple[OfflineSchedulerProcessId, ...]
    service_unavailable_dp_ranks: tuple[int, ...]
    unavailable_tp_groups: tuple[OfflineSchedulerProcessId, ...]
    incomplete_tp_groups: tuple[OfflineSchedulerProcessId, ...]
    node_dram_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", OfflineFailureScope(self.scope))
        object.__setattr__(
            self, "physical_gpus", tuple(sorted(set(self.physical_gpus)))
        )
        object.__setattr__(
            self,
            "scheduler_processes",
            tuple(sorted(set(self.scheduler_processes))),
        )
        object.__setattr__(
            self,
            "service_unavailable_dp_ranks",
            tuple(sorted(set(int(rank) for rank in self.service_unavailable_dp_ranks))),
        )
        object.__setattr__(
            self,
            "unavailable_tp_groups",
            tuple(sorted(set(self.unavailable_tp_groups))),
        )
        object.__setattr__(
            self,
            "incomplete_tp_groups",
            tuple(sorted(set(self.incomplete_tp_groups))),
        )
        object.__setattr__(
            self, "node_dram_ids", tuple(sorted(set(map(str, self.node_dram_ids))))
        )
        if not self.physical_gpus:
            raise ValueError("a resolved failure must target at least one GPU")
        if not self.scheduler_processes:
            raise ValueError("a resolved failure must affect a Scheduler process")
        if not self.service_unavailable_dp_ranks:
            raise ValueError("a resolved failure must make a DP replica unavailable")
        if not set(self.incomplete_tp_groups) <= set(self.unavailable_tp_groups):
            raise ValueError("incomplete TP groups must be unavailable")

    @property
    def physical_gpu_ids(self) -> tuple[str, ...]:
        return tuple(gpu.key for gpu in self.physical_gpus)

    @property
    def scheduler_process_ids(self) -> tuple[str, ...]:
        return tuple(process.key for process in self.scheduler_processes)

    @property
    def recovery_source_forbidden_groups(self) -> tuple[str, ...]:
        """Every touched group is unusable until an all-shard repair/reload."""

        return tuple(group.key for group in self.unavailable_tp_groups)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": OFFLINE_FAILURE_TARGET_SCHEMA,
            "failure_scope": self.scope.value,
            "physical_gpu_ids": list(self.physical_gpu_ids),
            "physical_gpus": [gpu.as_dict() for gpu in self.physical_gpus],
            "scheduler_process_ids": list(self.scheduler_process_ids),
            "scheduler_processes": [
                process.as_dict() for process in self.scheduler_processes
            ],
            "service_unavailable_dp_ranks": list(
                self.service_unavailable_dp_ranks
            ),
            "unavailable_tp_groups": [
                group.as_dict() for group in self.unavailable_tp_groups
            ],
            "incomplete_tp_groups": [
                group.as_dict() for group in self.incomplete_tp_groups
            ],
            "recovery_source_forbidden_groups": list(
                self.recovery_source_forbidden_groups
            ),
            "node_dram_ids": list(self.node_dram_ids),
            "repair_contract": {
                "targeted_gpu_volatile_state": "empty",
                "targeted_node_dram_volatile_state": (
                    "empty" if self.node_dram_ids else "not_targeted"
                ),
                "lost_kv_resurrection_forbidden": True,
            },
        }


class FailureTargetNotRepresentableError(ValueError):
    """The exact target exists but the current process adapter cannot inject it."""


def _all_groups(
    topology: OfflinePhysicalTopology,
) -> dict[OfflineSchedulerProcessId, tuple[PhysicalGpuId, ...]]:
    return {
        OfflineSchedulerProcessId(dp_rank, pp_rank): tuple(
            spec.gpu for spec in topology.group(dp_rank, pp_rank)
        )
        for dp_rank, pp_rank in topology.rank_groups
    }


def _only_value(name: str, values: tuple[int, ...]) -> int:
    if len(values) != 1:
        raise ValueError(f"{name} is required because the target is ambiguous")
    return values[0]


def resolve_offline_failure_target(
    topology: OfflinePhysicalTopology,
    *,
    failure_scope: OfflineFailureScope | str,
    dp_rank: int | None = None,
    pp_rank: int | None = None,
    tp_rank: int | None = None,
    node_id: str | None = None,
) -> ResolvedOfflineFailureTarget:
    """Resolve a logical scope into exact GPU/process/DRAM identifiers.

    ``pp_rank`` and ``tp_rank`` may be omitted only when the physical topology
    leaves exactly one choice.  This prevents a declaration such as
    ``gpu_1`` from accidentally meaning every GPU in DP rank 1.
    """

    scope = OfflineFailureScope(failure_scope)
    groups = _all_groups(topology)
    dp_values = tuple(sorted({group.dp_rank for group in groups}))
    known_nodes = {node.node_id for node in topology.nodes}

    if scope is OfflineFailureScope.NODE:
        if dp_rank is not None or pp_rank is not None or tp_rank is not None:
            raise ValueError("node failure forbids dp_rank/pp_rank/tp_rank")
        node = str(node_id or "")
        if node not in known_nodes:
            raise ValueError(f"unknown physical node {node!r}")
        selected = {
            gpu
            for physical_group in groups.values()
            for gpu in physical_group
            if gpu.node_id == node
        }
        if not selected:
            raise ValueError(f"node failure targets no GPU on {node!r}")
        node_drams = (node,)
    else:
        if node_id is not None:
            raise ValueError("node_id is valid only for node failure")
        if dp_rank is None:
            selected_dp = _only_value("dp_rank", dp_values)
        else:
            selected_dp = int(dp_rank)
        if selected_dp not in dp_values:
            raise ValueError(f"unknown physical DP rank {selected_dp}")
        dp_groups = tuple(group for group in groups if group.dp_rank == selected_dp)

        if scope is OfflineFailureScope.WORKER:
            if pp_rank is not None or tp_rank is not None:
                raise ValueError("worker failure forbids pp_rank/tp_rank")
            selected = {
                gpu for group in dp_groups for gpu in groups[group]
            }
        else:
            pp_values = tuple(sorted(group.pp_rank for group in dp_groups))
            selected_pp = (
                _only_value("pp_rank", pp_values)
                if pp_rank is None
                else int(pp_rank)
            )
            process = OfflineSchedulerProcessId(selected_dp, selected_pp)
            if process not in groups:
                raise ValueError(
                    f"unknown physical Scheduler process {process.key}"
                )
            physical_group = groups[process]
            if scope is OfflineFailureScope.PROCESS:
                if tp_rank is not None:
                    raise ValueError("process failure forbids tp_rank")
                selected = set(physical_group)
            else:
                tp_values = tuple(gpu.tp_rank for gpu in physical_group)
                selected_tp = (
                    _only_value("tp_rank", tp_values)
                    if tp_rank is None
                    else int(tp_rank)
                )
                matches = [gpu for gpu in physical_group if gpu.tp_rank == selected_tp]
                if len(matches) != 1:
                    raise ValueError(
                        f"unknown physical GPU dp{selected_dp}/pp{selected_pp}/"
                        f"tp{selected_tp}"
                    )
                selected = {matches[0]}
        node_drams = ()

    touched: list[OfflineSchedulerProcessId] = []
    incomplete: list[OfflineSchedulerProcessId] = []
    for process, physical_group in sorted(groups.items()):
        overlap = selected.intersection(physical_group)
        if not overlap:
            continue
        touched.append(process)
        if len(overlap) != len(physical_group):
            incomplete.append(process)

    return ResolvedOfflineFailureTarget(
        scope=scope,
        physical_gpus=tuple(selected),
        scheduler_processes=tuple(touched),
        service_unavailable_dp_ranks=tuple(
            sorted({process.dp_rank for process in touched})
        ),
        unavailable_tp_groups=tuple(touched),
        incomplete_tp_groups=tuple(incomplete),
        node_dram_ids=node_drams,
    )


def assert_current_cluster_representable(
    topology: OfflinePhysicalTopology,
    target: ResolvedOfflineFailureTarget,
) -> None:
    """Fail closed unless today's DP-wide restart produces the exact target.

    ``OfflineSchedulerCluster`` currently fences and restarts a whole DP rank.
    Worker failures are exact.  Node failures are exact only when every
    affected DP replica is wholly placed on that node.  Process/GPU scopes are
    accepted only for the degenerate PP=TP=1 case where their exact physical
    target is identical to the worker target.
    """

    groups = _all_groups(topology)
    target_gpus = set(target.physical_gpus)
    for dp_rank in target.service_unavailable_dp_ranks:
        whole_dp = {
            gpu
            for process, physical_group in groups.items()
            if process.dp_rank == dp_rank
            for gpu in physical_group
        }
        if not whole_dp <= target_gpus:
            raise FailureTargetNotRepresentableError(
                f"current DP-wide restart would widen {target.scope.value} failure "
                f"beyond exact GPU target for dp{dp_rank}"
            )
    if target.incomplete_tp_groups:
        names = ", ".join(group.key for group in target.incomplete_tp_groups)
        raise FailureTargetNotRepresentableError(
            "current symmetric TP snapshot cannot inject a partial TP-group "
            f"failure: {names}"
        )


__all__ = [
    "FailureTargetNotRepresentableError",
    "OFFLINE_FAILURE_TARGET_SCHEMA",
    "OfflineFailureScope",
    "OfflineSchedulerProcessId",
    "ResolvedOfflineFailureTarget",
    "assert_current_cluster_representable",
    "resolve_offline_failure_target",
]
