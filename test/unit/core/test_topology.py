import pytest

from sglsim.srt.distributed.parallel_state import RankStatus, Topology


@pytest.mark.parametrize(
    ("kwargs", "expected_name"),
    [
        ({"dp_size": 0}, "dp_size"),
        ({"tp_size": 0}, "tp_size"),
        ({"pp_size": 0}, "pp_size"),
    ],
)
def test_topology_rejects_non_positive_dimensions(kwargs, expected_name):
    with pytest.raises(ValueError, match=expected_name):
        Topology(**kwargs)


def test_topology_stores_public_dimension_attributes():
    topo = Topology(dp_size=2, tp_size=3, pp_size=4)

    assert topo.dp_size == 2
    assert topo.tp_size == 3
    assert topo.pp_size == 4


def test_topology_expands_dp_tp_pp_logical_ranks():
    topo = Topology(dp_size=2, tp_size=2, pp_size=1)

    assert topo.num_ranks == 4
    assert topo.rank(0).dp_rank == 0
    assert topo.rank(1).tp_rank == 1
    assert topo.rank(2).dp_rank == 1


def test_topology_can_mark_rank_busy_and_available():
    topo = Topology(dp_size=2)

    topo.mark_busy(1)
    assert topo.rank(1).status == RankStatus.BUSY

    topo.mark_available(1)
    assert topo.rank(1).status == RankStatus.AVAILABLE


@pytest.mark.parametrize("rank_id", [-1, 2])
def test_topology_rejects_rank_ids_outside_logical_range(rank_id):
    topo = Topology(dp_size=2)

    with pytest.raises(ValueError, match="rank_id"):
        topo.rank(rank_id)


def test_ranks_for_dp_returns_all_ranks_in_dp_group():
    topo = Topology(dp_size=2, tp_size=2, pp_size=2)

    ranks = topo.ranks_for_dp(1)

    assert [rank.rank_id for rank in ranks] == [4, 5, 6, 7]
    assert {rank.dp_rank for rank in ranks} == {1}
    assert {(rank.pp_rank, rank.tp_rank) for rank in ranks} == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }
