from enum import Enum


class CacheTier(str, Enum):
    L1 = "l1"
    L2 = "l2"
    L3 = "l3"


class TieredKVCache:
    def __init__(self, capacities: dict[CacheTier, int]):
        for tier, capacity in capacities.items():
            if capacity < 0:
                raise ValueError(f"{tier.value} capacity must be non-negative")

        self._capacities = dict(capacities)
        self._page_tiers: dict[int, CacheTier] = {}
        self._counts = {tier: 0 for tier in CacheTier}

    @property
    def page_tiers(self) -> dict[int, CacheTier]:
        return dict(self._page_tiers)

    def put(self, page_id: int, tier: CacheTier) -> None:
        if page_id in self._page_tiers:
            raise ValueError("page_id already exists in tiered KV cache")
        self._ensure_capacity(tier)

        self._page_tiers[page_id] = tier
        self._counts[tier] += 1

    def move(self, page_id: int, target_tier: CacheTier) -> None:
        current_tier = self._page_tiers.get(page_id)
        if current_tier is None:
            raise KeyError(page_id)
        if current_tier == target_tier:
            return
        self._ensure_capacity(target_tier)

        self._counts[current_tier] -= 1
        self._counts[target_tier] += 1
        self._page_tiers[page_id] = target_tier

    def tier_of(self, page_id: int) -> CacheTier | None:
        return self._page_tiers.get(page_id)

    def count(self, tier: CacheTier) -> int:
        return self._counts[tier]

    def free_slots(self, tier: CacheTier) -> int:
        capacity = self._capacities.get(tier, 0)
        return max(0, capacity - self._counts[tier])

    def remove(self, page_id: int) -> CacheTier | None:
        tier = self._page_tiers.pop(page_id, None)
        if tier is None:
            return None
        self._counts[tier] -= 1
        return tier

    def _ensure_capacity(self, tier: CacheTier) -> None:
        if self._counts[tier] >= self._capacities.get(tier, 0):
            raise RuntimeError(f"{tier.value} KV cache tier is full")
