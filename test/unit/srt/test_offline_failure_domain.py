from __future__ import annotations

import pytest

from sglsim.srt.simulation.global_des import PhysicalGpuId
from sglsim.srt.simulation.offline_failure_domain import (
    FailureTargetNotRepresentableError,
    OfflineFailureScope,
    assert_current_cluster_representable,
    resolve_offline_failure_target,
)
from sglsim.srt.simulation.offline_resource_bridge import (
    NodeDramSpec,
    OfflinePhysicalTopology,
    PhysicalGpuSpec,
)


def _topology(*, dp_size: int = 2, pp_size: int = 2, tp_size: int = 2):
    gpus: list[PhysicalGpuSpec] = []
    linear = 0
    for dp_rank in range(dp_size):
        for pp_rank in range(pp_size):
            for tp_rank in range(tp_size):
                node_index = linear // 4
                local_gpu = linear % 4
                gpus.append(
                    PhysicalGpuSpec(
                        PhysicalGpuId(
                            f"node{node_index}",
                            local_gpu,
                            dp_rank,
                            pp_rank,
                            tp_rank,
                        ),
                        hbm_capacity_bytes=1000,
                        weights_bytes=100,
                        runtime_bytes=10,
                    )
                )
                linear += 1
    node_ids = sorted({spec.gpu.node_id for spec in gpus})
    return OfflinePhysicalTopology(
        gpus=tuple(gpus),
        nodes=tuple(NodeDramSpec(node, 10_000) for node in node_ids),
    )


def test_worker_resolves_all_processes_and_gpus_of_one_dp() -> None:
    topology = _topology()
    target = resolve_offline_failure_target(
        topology,
        failure_scope="worker",
        dp_rank=1,
    )

    assert target.scope is OfflineFailureScope.WORKER
    assert target.scheduler_process_ids == ("dp1/pp0", "dp1/pp1")
    assert len(target.physical_gpus) == 4
    assert {gpu.dp_rank for gpu in target.physical_gpus} == {1}
    assert target.service_unavailable_dp_ranks == (1,)
    assert target.incomplete_tp_groups == ()
    assert target.node_dram_ids == ()
    assert_current_cluster_representable(topology, target)


def test_process_is_one_dp_pp_process_and_its_complete_tp_group() -> None:
    topology = _topology()
    target = resolve_offline_failure_target(
        topology,
        failure_scope="process",
        dp_rank=0,
        pp_rank=1,
    )

    assert target.scheduler_process_ids == ("dp0/pp1",)
    assert {(gpu.pp_rank, gpu.tp_rank) for gpu in target.physical_gpus} == {
        (1, 0),
        (1, 1),
    }
    assert target.incomplete_tp_groups == ()
    assert target.recovery_source_forbidden_groups == ("dp0/pp1",)
    with pytest.raises(
        FailureTargetNotRepresentableError,
        match="DP-wide restart would widen process failure",
    ):
        assert_current_cluster_representable(topology, target)


def test_gpu_is_one_physical_id_and_marks_tp_group_incomplete() -> None:
    topology = _topology()
    target = resolve_offline_failure_target(
        topology,
        failure_scope="gpu",
        dp_rank=0,
        pp_rank=1,
        tp_rank=1,
    )

    assert len(target.physical_gpus) == 1
    gpu = target.physical_gpus[0]
    assert (gpu.dp_rank, gpu.pp_rank, gpu.tp_rank) == (0, 1, 1)
    assert target.scheduler_process_ids == ("dp0/pp1",)
    assert [group.key for group in target.incomplete_tp_groups] == ["dp0/pp1"]
    assert target.as_dict()["physical_gpu_ids"] == [gpu.key]
    with pytest.raises(FailureTargetNotRepresentableError):
        assert_current_cluster_representable(topology, target)


def test_node_resolves_local_gpu_and_dram_ids_not_whole_dp_by_name() -> None:
    topology = _topology()
    target = resolve_offline_failure_target(
        topology,
        failure_scope="node",
        node_id="node0",
    )

    assert target.node_dram_ids == ("node0",)
    assert len(target.physical_gpus) == 4
    assert {gpu.node_id for gpu in target.physical_gpus} == {"node0"}
    assert target.scheduler_process_ids == ("dp0/pp0", "dp0/pp1")
    assert target.service_unavailable_dp_ranks == (0,)
    assert_current_cluster_representable(topology, target)


def test_cross_node_tp_group_node_failure_is_resolved_but_not_representable() -> None:
    topology = OfflinePhysicalTopology(
        gpus=(
            PhysicalGpuSpec(
                PhysicalGpuId("node0", 0, 0, 0, 0), 1000
            ),
            PhysicalGpuSpec(
                PhysicalGpuId("node1", 0, 0, 0, 1), 1000
            ),
            PhysicalGpuSpec(
                PhysicalGpuId("node2", 0, 1, 0, 0), 1000
            ),
            PhysicalGpuSpec(
                PhysicalGpuId("node2", 1, 1, 0, 1), 1000
            ),
        ),
        nodes=tuple(NodeDramSpec(f"node{index}", 10_000) for index in range(3)),
    )
    target = resolve_offline_failure_target(
        topology,
        failure_scope="node",
        node_id="node0",
    )

    assert [group.key for group in target.incomplete_tp_groups] == ["dp0/pp0"]
    with pytest.raises(FailureTargetNotRepresentableError):
        assert_current_cluster_representable(topology, target)


def test_degenerate_gpu_and_process_are_exact_worker_aliases() -> None:
    topology = _topology(dp_size=2, pp_size=1, tp_size=1)
    for scope in ("gpu", "process"):
        target = resolve_offline_failure_target(
            topology,
            failure_scope=scope,
            dp_rank=0,
        )
        assert target.physical_gpu_ids == (
            "node0/gpu0/dp0/pp0/tp0",
        )
        assert_current_cluster_representable(topology, target)


def test_ambiguous_process_or_gpu_target_fails_closed() -> None:
    topology = _topology()
    with pytest.raises(ValueError, match="pp_rank is required"):
        resolve_offline_failure_target(
            topology,
            failure_scope="process",
            dp_rank=0,
        )
    with pytest.raises(ValueError, match="tp_rank is required"):
        resolve_offline_failure_target(
            topology,
            failure_scope="gpu",
            dp_rank=0,
            pp_rank=0,
        )
    with pytest.raises(ValueError, match="node failure forbids"):
        resolve_offline_failure_target(
            topology,
            failure_scope="node",
            node_id="node0",
            dp_rank=0,
        )
