from dataclasses import dataclass
from enum import Enum


class RankStatus(Enum):
    AVAILABLE = "available"
    BUSY = "busy"


@dataclass
class Rank:
    rank_id: int
    dp_rank: int
    tp_rank: int
    pp_rank: int
    status: RankStatus = RankStatus.AVAILABLE


class Topology:
    def __init__(self, dp_size: int = 1, tp_size: int = 1, pp_size: int = 1):
        for name, value in (
            ("dp_size", dp_size),
            ("tp_size", tp_size),
            ("pp_size", pp_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.pp_size = pp_size
        self._ranks: list[Rank] = []
        rank_id = 0
        for dp_rank in range(self.dp_size):
            for pp_rank in range(self.pp_size):
                for tp_rank in range(self.tp_size):
                    self._ranks.append(
                        Rank(
                            rank_id=rank_id,
                            dp_rank=dp_rank,
                            tp_rank=tp_rank,
                            pp_rank=pp_rank,
                        )
                    )
                    rank_id += 1

    @property
    def num_ranks(self) -> int:
        return len(self._ranks)

    def rank(self, rank_id: int) -> Rank:
        if rank_id < 0 or rank_id >= len(self._ranks):
            raise ValueError(f"rank_id must be between 0 and {len(self._ranks) - 1}")
        return self._ranks[rank_id]

    def mark_busy(self, rank_id: int) -> None:
        self.rank(rank_id).status = RankStatus.BUSY

    def mark_available(self, rank_id: int) -> None:
        self.rank(rank_id).status = RankStatus.AVAILABLE

    def ranks_for_dp(self, dp_rank: int) -> list[Rank]:
        if dp_rank < 0 or dp_rank >= self.dp_size:
            raise ValueError(f"dp_rank must be between 0 and {self.dp_size - 1}")
        return [rank for rank in self._ranks if rank.dp_rank == dp_rank]
